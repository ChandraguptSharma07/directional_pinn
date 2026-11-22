"""Reference-free directions: the planned test of docs/reference_free_plan.md.

For every run and horizon the script compares three versions of the leading
direction:

* oracle s_1: from the tangent propagator along the reference solution;
* P: from the tangent propagator along the PINN's own trajectory, with a
  cubic spline through its outputs on the 0.002 grid (primary estimate);
* F: from expm(J(x_hat(t)) dt), the Jacobian frozen at the PINN state
  (secondary estimate).

It fits the dial-study baseline B (log R, tau, tau^2) and, for each direction,
B + log(|r . u| + floor) to the log created error. Evaluation is
leave-one-window-out with whole-window bootstrap intervals, and 20 random
directions serve as a control.

Data sets:
* fresh: windows 9-16, which decide the verdict;
* development: the dial-study windows 1-8 (descriptive);
* lorenz: the October Lorenz-63 runs (descriptive).

    python experiments/27_reference_free_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.metrics import bootstrap_delta_r_squared, cluster_sums, leave_one_group_out, ols_fit  # noqa: E402
from src.analysis.plots import COLORS, apply_style, save_figure  # noqa: E402
from src.dynamics.lyapunov import singular_directions  # noqa: E402
from src.dynamics.tangent_dynamics import (  # noqa: E402
    frozen_jacobian_propagators,
    pinn_path_step_propagators,
    propagators_over,
)
from src.systems.dial_systems import dial_settings  # noqa: E402
from src.systems.lorenz63 import LorenzParams, lorenz_jacobian  # noqa: E402
from src.utils.reproducibility import save_json, write_run_record  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (0.05, 0.1, 0.25, 0.5)
PRIMARY = 0.1
STRIDE, N_RANDOM, N_BOOT, FLOOR, TINY = 5, 20, 2000, 1e-6, 1e-12
BASE = ["log_R", "tau", "tau2"]
DIRECTIONS = ("oracle", "P", "F")
FRESH_SETTINGS = ("saddle_k1", "saddle_k64", "davis_skodje_g18", "davis_skodje_g66")
REPLICATION_SETTINGS = ("saddle_k64", "davis_skodje_g18", "davis_skodje_g66")
REFERENCE_FREE_SETTINGS = ("davis_skodje_g18", "davis_skodje_g66")


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(runs_dir.glob("w*_s*")):
        if not (run_dir / "metrics.json").exists():
            continue
        summary = json.loads((run_dir / "metrics.json").read_text())
        trajectory, residuals = np.load(run_dir / "trajectory.npz"), np.load(run_dir / "residuals.npz")
        runs.append({"name": run_dir.name, "window": int(summary["window"]), "seed": int(summary["seed"]),
                     "finite": bool(summary["finite"]), "summary": summary, "t": trajectory["t"],
                     "error": trajectory["error"], "pinn": trajectory["pinn"], "residual": residuals["residual"]})
    return runs


def line_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle in degrees between the lines spanned by unit vectors u and v (rows)."""
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(u * v, axis=1)), 0.0, 1.0)))


def build_samples(runs: list[dict], directions: dict, jacobian) -> tuple[dict, dict]:
    tables = {dt: [] for dt in HORIZONS}
    dropped = {}
    for run_index, run in enumerate(runs):
        if not run["finite"]:
            continue
        D = directions[run["window"]]
        t, e, r, x_hat = run["t"], run["error"], run["residual"], run["pinn"]
        n, h = len(t), t[1] - t[0]
        if not np.allclose(D["t"][:n], t, atol=1e-9):
            raise ValueError(f"{run['name']}: grid differs from the direction data")
        steps_ref = D["step_propagators"][: n - 1]
        steps_pinn = pinn_path_step_propagators(t, x_hat, jacobian)  # reference-free
        R_all = np.linalg.norm(r, axis=1)
        floor = FLOOR * np.median(R_all)
        for k, dt in enumerate(HORIZONS):
            m = int(round(dt / h))
            j = np.arange(0, n - m, STRIDE)
            if len(j) < 10:
                continue
            phi_pinn = propagators_over(steps_pinn, m)[j]
            frozen = frozen_jacobian_propagators(x_hat[j], jacobian, dt)
            # Samples whose reference-free propagator is not finite (an inaccurate PINN path near a
            # singularity of the equations) are dropped from every model, and counted.
            keep = (np.all(np.isfinite(phi_pinn), axis=(1, 2)) & np.all(np.isfinite(frozen), axis=(1, 2))
                    & np.all(np.abs(phi_pinn) < 1e150, axis=(1, 2)) & np.all(np.abs(frozen) < 1e150, axis=(1, 2)))
            dropped[dt] = dropped.get(dt, 0) + int((~keep).sum())
            if not keep.any():
                continue
            j, phi_pinn, frozen = j[keep], phi_pinn[keep], frozen[keep]
            phi = propagators_over(steps_ref, m)[j]
            u = {"oracle": D[f"V_{dt:g}"][j][:, :, 0],
                 "P": singular_directions(phi_pinn)[1][:, :, 0],
                 "F": singular_directions(frozen)[1][:, :, 0]}
            rj, R = r[j], R_all[j]
            created = e[j + m] - np.einsum("jab,jb->ja", phi, e[j])
            row = {"run": np.full(len(j), run_index), "window": np.full(len(j), run["window"]),
                   "t": t[j], "tau": t[j] / t[-1], "tau2": (t[j] / t[-1]) ** 2, "log_R": np.log(R),
                   "log_N": np.log(np.linalg.norm(created, axis=1) + TINY),
                   "log_E": np.log(np.linalg.norm(e[j + m], axis=1) + TINY),
                   "angle_P": line_angle(u["P"], u["oracle"]), "angle_F": line_angle(u["F"], u["oracle"])}
            for name, vec in u.items():
                row[f"log_Ru_{name}"] = np.log(np.abs(np.einsum("ja,ja->j", rj, vec)) + floor)
            for q in range(N_RANDOM):
                g = np.random.default_rng([q, run_index, k]).normal(size=rj.shape)
                g /= np.linalg.norm(g, axis=1, keepdims=True)
                row[f"rand_{q}"] = np.log(np.abs(np.einsum("ja,ja->j", rj, g)) + floor)
            tables[dt].append(row)
    samples = {dt: {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
               for dt, rows in tables.items() if rows}
    return samples, {f"{dt:g}": n for dt, n in dropped.items()}


def heldout(S: dict, columns: list[str], target: str) -> np.ndarray:
    return leave_one_group_out(np.column_stack([S[c] for c in columns]), S[target], S["window"])


def analyse(S: dict, target: str = "log_N") -> dict:
    y = S[target]
    base = heldout(S, BASE, target)
    out, preds = {}, {}
    for name in DIRECTIONS:
        columns = BASE + [f"log_Ru_{name}"]
        preds[name] = heldout(S, columns, target)
        stats = bootstrap_delta_r_squared(y, base, preds[name], S["window"], n_boot=N_BOOT, seed=0)
        ids, sums = cluster_sums(y, [base, preds[name]], S["window"])
        out[name] = {**stats, "coefficient": float(ols_fit(np.column_stack([S[c] for c in columns]), y)[-1]),
                     "windows_improved": int(np.sum(1.0 - sums[:, 4] / sums[:, 3] > 0))}
    random_gains = []
    for q in range(N_RANDOM):
        S["_c"] = S[f"rand_{q}"]
        random_gains.append(float(bootstrap_delta_r_squared(y, base, heldout(S, BASE + ["_c"], target),
                                                            S["window"], n_boot=1)["delta"]))
    S.pop("_c", None)
    out["random_p95"] = float(np.quantile(random_gains, 0.95))
    for name in ("P", "F"):
        paired = bootstrap_delta_r_squared(y, preds["oracle"], preds[name], S["window"], n_boot=N_BOOT, seed=1)
        out[f"{name}_minus_oracle"] = {"delta": paired["delta"], "ci_low": paired["ci_low"],
                                       "ci_high": paired["ci_high"]}
        out[f"{name}_recovery"] = (out[name]["delta"] / out["oracle"]["delta"]
                                   if out["oracle"]["delta"] > 0 else None)
    out["angles"] = {f"{name}_p50_p90": np.quantile(S[f"angle_{name}"], [0.5, 0.9]).tolist() for name in ("P", "F")}
    out["n_samples"] = int(len(y))
    return out


def works(res: dict, name: str) -> bool:
    s = res[name]
    return bool(s["delta"] >= 0.05 and s["ci_low"] > 0 and s["coefficient"] > 0 and s["delta"] > res["random_p95"])


def run_set(runs_root: Path, reference_root: Path, settings: list) -> dict:
    results = {}
    for system in settings:
        runs = load_runs(runs_root / system.name / "runs")
        if not runs:
            continue
        directions = {w: dict(np.load(reference_root / system.name / "data" / f"window_{w}.npz"))
                      for w in sorted({r["window"] for r in runs})}
        samples, dropped = build_samples(runs, directions, system.jacobian)
        results[system.name] = {"dial": system.dial, "n_runs": len(runs),
                                "n_finite": sum(r["finite"] for r in runs), "samples_dropped": dropped,
                                "N": {f"{dt:g}": analyse(S, "log_N") for dt, S in samples.items()},
                                "E": {f"{dt:g}": analyse(S, "log_E") for dt, S in samples.items()}}
        p = results[system.name]["N"][f"{PRIMARY:g}"]
        print(f"  {system.name:18s} oracle {p['oracle']['delta']:+.3f} [{p['oracle']['ci_low']:+.3f},"
              f"{p['oracle']['ci_high']:+.3f}]  P {p['P']['delta']:+.3f} [{p['P']['ci_low']:+.3f},"
              f"{p['P']['ci_high']:+.3f}]  F {p['F']['delta']:+.3f}  random p95 {p['random_p95']:+.4f}  "
              f"angle P median {p['angles']['P_p50_p90'][0]:.2f} deg", flush=True)
    return results


def gains_figure(fresh: dict, development: dict, out_dir: Path) -> None:
    """Primary-horizon gains of the oracle, P and F directions, on fresh and development windows."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), sharey=True)
    styles = {"oracle": (COLORS["ink"], "D", "oracle s₁ (needs the true solution)"),
              "P": (COLORS["stable"], "o", "P: from the PINN's own path (reference-free)"),
              "F": (COLORS["third"], "s", "F: Jacobian frozen at the PINN state (reference-free)")}
    for ax, (title, data) in zip(axes, [("fresh starting points 9–16 (decides the verdict)", fresh),
                                        ("dial-study starting points 1–8 (descriptive)", development)]):
        names = list(data)
        x = np.arange(len(names))
        for offset, name in zip((-0.2, 0.0, 0.2), DIRECTIONS):
            color, marker, label = styles[name]
            stats = [data[n]["N"][f"{PRIMARY:g}"][name] for n in names]
            y = np.array([s["delta"] for s in stats])
            err = np.array([[s["delta"] - s["ci_low"] for s in stats], [s["ci_high"] - s["delta"] for s in stats]])
            ax.errorbar(x + offset, y, yerr=err, color=color, marker=marker, markersize=6, linestyle="none",
                        capsize=2.5, elinewidth=0.9, label=label,
                        markerfacecolor="white" if name == "oracle" else color)
        ax.axhline(0.05, color=COLORS["muted"], linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color=COLORS["axis"], linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n.replace('davis_skodje_g', 'D, γ=').replace('saddle_k', 'L, κ=')}" for n in names],
                           rotation=0, fontsize=8.5)
        ax.set_title(title)
    axes[0].set_ylabel("held-out gain in R² (created error, dt = 0.1)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Does a direction computed without the true solution still predict the error a residual creates?",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig01_reference_free_gains")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/reference_free/analysis")
    args = parser.parse_args()
    start = time.perf_counter()
    apply_style()
    by_name = {s.name: s for s in dial_settings()}
    print("fresh windows 9-16 (decision):")
    fresh = run_set(REPO_ROOT / "results/reference_free", REPO_ROOT / "results/reference_free/reference",
                    [by_name[n] for n in FRESH_SETTINGS])
    print("dial-study windows 1-8 (descriptive):")
    development = run_set(REPO_ROOT / "results/dial_study", REPO_ROOT / "results/dial_study/reference",
                          list(dial_settings()))
    print("Lorenz-63 October runs (descriptive):")
    params = LorenzParams()
    lorenz_runs = load_runs(REPO_ROOT / "results/pinn_runs/runs")
    lorenz_dirs = {w: dict(np.load(REPO_ROOT / "results/lorenz_lyapunov/data" / f"lyapunov_window_{w}.npz"))
                   for w in sorted({r["window"] for r in lorenz_runs})}
    lorenz_samples, lorenz_dropped = build_samples(lorenz_runs, lorenz_dirs, lambda X: lorenz_jacobian(X, params))
    lorenz = {"samples_dropped": lorenz_dropped,
              "N": {f"{dt:g}": analyse(S, "log_N") for dt, S in lorenz_samples.items()},
              "E": {f"{dt:g}": analyse(S, "log_E") for dt, S in lorenz_samples.items()}}
    p = lorenz["N"][f"{PRIMARY:g}"]
    print(f"  lorenz63           oracle {p['oracle']['delta']:+.3f}  P {p['P']['delta']:+.3f}  F {p['F']['delta']:+.3f}"
          f"  angle P median {p['angles']['P_p50_p90'][0]:.2f} deg, p90 {p['angles']['P_p50_p90'][1]:.2f} deg")

    primary = {n: fresh[n]["N"][f"{PRIMARY:g}"] for n in fresh}
    replication = {n: works(primary[n], "oracle") for n in REPLICATION_SETTINGS if n in primary}
    control_fails = primary["saddle_k1"]["oracle"]["delta"] < 0.02 if "saddle_k1" in primary else None
    decision = {}
    for name in ("P", "F"):
        decision[name] = {n: bool(works(primary[n], name)
                                  and primary[n][name]["delta"] >= 0.5 * primary[n]["oracle"]["delta"])
                          for n in REFERENCE_FREE_SETTINGS if n in primary}
    count = sum(decision["P"].values())
    verdict = ({2: "supported", 1: "partly supported", 0: "not supported"}[count]
               if len(decision["P"]) == 2 else "incomplete")
    metrics = {"plan": "docs/reference_free_plan.md", "primary_horizon": PRIMARY, "fresh": fresh,
               "development": development, "lorenz": lorenz,
               "decision": {"A_oracle_replicates": replication, "A_low_gap_control_fails": control_fails,
                            "B_reference_free_P_works": decision["P"], "secondary_F_works": decision["F"],
                            "verdict_P": verdict}}
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.out / "metrics.json")
    if fresh:
        gains_figure(fresh, {n: development[n] for n in FRESH_SETTINGS if n in development}, args.out / "figures")
    write_run_record(args.out, {"script": "experiments/27_reference_free_analysis.py"}, REPO_ROOT,
                     time.perf_counter() - start)
    print("A, oracle replicates on fresh windows:", replication, "| low-gap control fails:", control_fails)
    print("B, reference-free P works:", decision["P"], "| F:", decision["F"])
    print("verdict for P:", verdict)


if __name__ == "__main__":
    main()
