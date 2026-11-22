"""Dial study, step 3: the planned analysis (docs/dial_study_plan.md, sections 6-10).

For each dial setting the script:
* builds samples every 0.01 time units for horizons dt in {0.05, 0.1, 0.25, 0.5};
* fits the baseline B (log R, tau, tau^2) and the direction models M1-M4 to
  the log created error N (primary target) and the log future error E;
* evaluates them leave-one-window-out, with whole-window bootstrap intervals;
* recomputes the M2 gain with random directions (and time-shuffled s_1 for
  family D) as controls;
* applies the decision rules of section 9.

The October Lorenz set (results/pinn_runs, T = 1) goes through the same code
as a descriptive low-gap reference point.

    python experiments/26_dial_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.metrics import bootstrap_delta_r_squared, cluster_sums, leave_one_group_out, ols_fit  # noqa: E402
from src.analysis.plots import COLORS, apply_style, save_figure  # noqa: E402
from src.dynamics.tangent_dynamics import integrated_propagators, propagators_over  # noqa: E402
from src.systems.dial_systems import DavisSkodje, StiffSaddle, dial_settings  # noqa: E402
from src.utils.reproducibility import save_json, write_run_record  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (0.05, 0.1, 0.25, 0.5)
PRIMARY_HORIZON = 0.1
STRIDE = 5  # every 5th grid point: spacing 0.01
N_RANDOM, N_SHUFFLE, N_BOOT = 20, 20, 2000
FLOOR, TINY = 1e-6, 1e-12
WORKS_GAIN, FAILS_GAIN, KENDALL_MIN = 0.05, 0.02, 0.66
BASE = ["log_R", "tau", "tau2"]
MODELS = {"M2": BASE + ["log_Ru"], "M1": BASE + ["A"], "M3": BASE + ["log_amp"], "M4": BASE + ["log_K"]}
TARGETS = {"N": "log_N", "E": "log_E"}
FAMILY_COLOR = {"L": COLORS["stable"], "D": COLORS["third"]}
FAMILY_LABEL = {"L": "stiff linear saddle (L)", "D": "Davis–Skodje (D)"}
FAMILY_MARKER = {"L": "o", "D": "s"}


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(runs_dir.glob("w*_s*")):
        summary = json.loads((run_dir / "metrics.json").read_text())
        trajectory, residuals = np.load(run_dir / "trajectory.npz"), np.load(run_dir / "residuals.npz")
        runs.append({"name": run_dir.name, "window": int(summary["window"]), "seed": int(summary["seed"]),
                     "finite": bool(summary["finite"]), "summary": summary, "t": trajectory["t"],
                     "error": trajectory["error"], "residual": residuals["residual"]})
    return runs


def build_samples(runs: list[dict], directions: dict, shuffle: bool) -> tuple[dict, list[str]]:
    """Samples per horizon: arrays keyed by feature name (plan section 6)."""
    tables = {dt: [] for dt in HORIZONS}
    excluded = []
    for run_index, run in enumerate(runs):
        if not run["finite"]:
            excluded.append(run["name"])
            continue
        D = directions[run["window"]]
        t, e, r = run["t"], run["error"], run["residual"]
        n, h = len(t), t[1] - t[0]
        if not np.allclose(D["t"][:n], t, atol=1e-9):
            raise ValueError(f"{run['name']}: time grid differs from the direction data")
        steps = D["step_propagators"][: n - 1]
        R_all = np.linalg.norm(r, axis=1)
        floor = FLOOR * np.median(R_all)
        unit = r / R_all[:, None]
        for k, dt in enumerate(HORIZONS):
            m = int(round(dt / h))
            j = np.arange(0, n - m, STRIDE)
            if len(j) < 10:
                continue
            phi = propagators_over(steps, m)[j]
            K = D[f"K_{dt:g}"][j] if f"K_{dt:g}" in D else integrated_propagators(steps, m, h)[j]
            s1 = D[f"V_{dt:g}"][j][:, :, 0]
            sigma = D[f"sigma_{dt:g}"][j]
            rj, R = r[j], R_all[j]
            created = e[j + m] - np.einsum("jab,jb->ja", phi, e[j])
            Ru = np.abs(np.einsum("ja,ja->j", rj, s1))
            row = {
                "run": np.full(len(j), run_index), "window": np.full(len(j), run["window"]),
                "seed": np.full(len(j), run["seed"]), "t": t[j], "tau": t[j] / t[-1], "tau2": (t[j] / t[-1]) ** 2,
                "log_N": np.log(np.linalg.norm(created, axis=1) + TINY),
                "log_E": np.log(np.linalg.norm(e[j + m], axis=1) + TINY),
                "log_R": np.log(R), "A": Ru / R, "log_Ru": np.log(Ru + floor),
                "log_amp": np.log(np.linalg.norm(np.einsum("jab,jb->ja", phi, rj), axis=1) / R),
                "log_K": np.log(np.linalg.norm(np.einsum("jab,jb->ja", K, rj), axis=1) / R),
                "sigma_ratio": sigma[:, 0] / sigma[:, 1],
                "persistence": np.sum(unit[j] * unit[j + m], axis=1),
            }
            for q in range(N_RANDOM):
                rng = np.random.default_rng([q, run_index, k])
                u = rng.normal(size=rj.shape)
                u /= np.linalg.norm(u, axis=1, keepdims=True)
                row[f"rand_{q}"] = np.log(np.abs(np.einsum("ja,ja->j", rj, u)) + floor)
            if shuffle:
                for q in range(N_SHUFFLE):
                    perm = np.random.default_rng([1000 + q, run_index, k]).permutation(len(j))
                    row[f"shuf_{q}"] = np.log(np.abs(np.einsum("ja,ja->j", rj, s1[perm])) + floor)
            tables[dt].append(row)
    samples = {dt: {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
               for dt, rows in tables.items() if rows}
    return samples, excluded


def heldout(S: dict, columns: list[str], target: str) -> np.ndarray:
    X = np.column_stack([S[c] for c in columns])
    return leave_one_group_out(X, S[target], S["window"])


def analyse(S: dict, shuffle: bool) -> dict:
    """All models and controls for one setting and horizon."""
    out = {"n_samples": int(len(S["log_R"])), "n_runs": int(len(np.unique(S["run"])))}
    for target_name, target in TARGETS.items():
        y = S[target]
        base = heldout(S, BASE, target)
        res = {}
        for model, columns in MODELS.items():
            pred = heldout(S, columns, target)
            stats = bootstrap_delta_r_squared(y, base, pred, S["window"], n_boot=N_BOOT, seed=0)
            coefficient = ols_fit(np.column_stack([S[c] for c in columns]), y)[-1]
            ids, sums = cluster_sums(y, [base, pred], S["window"])
            per_window = 1.0 - sums[:, 4] / sums[:, 3]
            res[model] = {**stats, "coefficient": float(coefficient),
                          "per_window_sse_reduction": dict(zip(map(str, ids), map(float, per_window))),
                          "windows_improved": int(np.sum(per_window > 0))}
        controls = {"random": [], "shuffled": []}
        for q in range(N_RANDOM):
            S[f"_c"] = S[f"rand_{q}"]
            controls["random"].append(float(bootstrap_delta_r_squared(
                y, base, heldout(S, BASE + ["_c"], target), S["window"], n_boot=1)["delta"]))
        if shuffle:
            for q in range(N_SHUFFLE):
                S["_c"] = S[f"shuf_{q}"]
                controls["shuffled"].append(float(bootstrap_delta_r_squared(
                    y, base, heldout(S, BASE + ["_c"], target), S["window"], n_boot=1)["delta"]))
        S.pop("_c", None)
        res["controls"] = {
            "random_gains": controls["random"], "random_p95": float(np.quantile(controls["random"], 0.95)),
            "shuffled_gains": controls["shuffled"],
            "shuffled_p95": float(np.quantile(controls["shuffled"], 0.95)) if shuffle else None,
        }
        out[target_name] = res
    out["descriptives"] = {
        "alignment_p10_p50_p90": np.quantile(S["A"], [0.1, 0.5, 0.9]).tolist(),
        "sigma_ratio_p10_p50_p90": np.quantile(S["sigma_ratio"], [0.1, 0.5, 0.9]).tolist(),
        "residual_persistence_mean": float(np.mean(S["persistence"])),
        "spearman_logN_logR": float(spearmanr(S["log_R"], S["log_N"]).statistic),
        "spearman_logN_logRu": float(spearmanr(S["log_Ru"], S["log_N"]).statistic),
    }
    return out


def setting_verdict(primary: dict, shuffle: bool) -> str:
    m2, controls = primary["N"]["M2"], primary["N"]["controls"]
    works = (m2["delta"] >= WORKS_GAIN and m2["ci_low"] > 0 and m2["coefficient"] > 0
             and m2["delta"] > controls["random_p95"] and (not shuffle or m2["delta"] > controls["shuffled_p95"]))
    if works:
        return "works"
    return "fails" if m2["delta"] < FAILS_GAIN else "inconclusive"


def accuracy(runs: list[dict]) -> dict:
    finite = [r for r in runs if r["finite"]]
    rel = [r["summary"]["max_error"] / r["summary"].get("max_reference_norm", np.nan) for r in finite]
    return {"n_runs": len(runs), "n_finite": len(finite),
            "median_of_run_median_errors": float(np.median([r["summary"]["median_error"] for r in finite])),
            "median_of_relative_max_errors": float(np.nanmedian(rel)) if rel else None,
            "worst_relative_max_error": float(np.nanmax(rel)) if rel else None}


def scatter_figure(samples_by_setting: dict, family: str, out_dir: Path) -> None:
    """Created error against residual size and against its component along s_1, lowest vs highest dial."""
    names = [s.name for s in dial_settings() if s.to_dict()["family"] == family]
    picks = [names[0], names[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharey="row")
    for row, name in enumerate(picks):
        S = samples_by_setting[name][PRIMARY_HORIZON]
        idx = np.random.default_rng(0).choice(len(S["log_N"]), size=min(4000, len(S["log_N"])), replace=False)
        for col, (key, label) in enumerate([("log_R", "residual size  ||r(t)||"),
                                            ("log_Ru", "residual along the leading direction  |r·s₁|")]):
            ax = axes[row, col]
            ax.scatter(np.exp(S[key][idx]), np.exp(S["log_N"][idx]), s=4, alpha=0.25, color=COLORS["ink_secondary"],
                       linewidths=0, rasterized=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            rho = spearmanr(S[key], S["log_N"]).statistic
            dial = next(s.dial for s in dial_settings() if s.name == name)
            ax.set_title(f"dial {dial:g}: rank correlation {rho:.2f}")
            if row == 1:
                ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel("error created over the next 0.1")
    n_runs = len(np.unique(samples_by_setting[picks[0]][PRIMARY_HORIZON]["run"]))
    fig.suptitle(f"{FAMILY_LABEL[family]}: what predicts the error a residual creates ({n_runs} runs, dt = 0.1)",
                 x=0.02, ha="left", fontsize=10.5)
    fig.tight_layout()
    save_figure(fig, out_dir, f"fig03_scatter_{family}")


def dial_figure(results: dict, lorenz: dict | None, out_dir: Path) -> None:
    """Direction gain (M2 over B, target N) against the stretching ratio, one panel per horizon."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(12.0, 3.6), sharey=True)
    for ax, dt in zip(axes, HORIZONS):
        for family in ("L", "D"):
            names = [s.name for s in dial_settings() if s.to_dict()["family"] == family]
            x = [np.log(results[n]["horizons"][f"{dt:g}"]["descriptives"]["sigma_ratio_p10_p50_p90"][1]) / dt
                 for n in names]
            stats = [results[n]["horizons"][f"{dt:g}"]["N"]["M2"] for n in names]
            y = np.array([s["delta"] for s in stats])
            err = np.array([[s["delta"] - s["ci_low"] for s in stats], [s["ci_high"] - s["delta"] for s in stats]])
            ax.errorbar(x, y, yerr=err, color=FAMILY_COLOR[family], marker=FAMILY_MARKER[family], markersize=6,
                        linewidth=1.6, capsize=2.5, elinewidth=0.9, label=FAMILY_LABEL[family])
        if lorenz is not None and f"{dt:g}" in lorenz["horizons"]:
            h = lorenz["horizons"][f"{dt:g}"]
            s = h["N"]["M2"]
            ax.errorbar([np.log(h["descriptives"]["sigma_ratio_p10_p50_p90"][1]) / dt], [s["delta"]],
                        yerr=[[s["delta"] - s["ci_low"]], [s["ci_high"] - s["delta"]]], color=COLORS["ink"],
                        marker="D", markersize=6, markerfacecolor="white", linestyle="none", capsize=2.5,
                        elinewidth=0.9, label="Lorenz-63, October runs")
        ax.axhline(WORKS_GAIN, color=COLORS["muted"], linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color=COLORS["axis"], linewidth=0.8)
        ax.set_xscale("log")
        ax.set_title(f"horizon dt = {dt:g}" + ("  (primary)" if dt == PRIMARY_HORIZON else ""))
        ax.set_xlabel("rate gap  ln(σ₁/σ₂)/dt  (median)")
    axes[0].set_ylabel("held-out gain in R² from direction")
    axes[0].text(axes[0].get_xlim()[0], WORKS_GAIN, " planned 'works' threshold", va="bottom", fontsize=8,
                 color=COLORS["ink_secondary"])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Does the residual's direction predict the error it creates? Gain of model M2 over the baseline",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig01_dial")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, default=REPO_ROOT / "results/dial_study")
    parser.add_argument("--lorenz-runs", type=Path, default=REPO_ROOT / "results/pinn_runs/runs")
    parser.add_argument("--lorenz-lyapunov", type=Path, default=REPO_ROOT / "results/lorenz_lyapunov/data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/dial_study/analysis")
    args = parser.parse_args()
    start = time.perf_counter()
    apply_style()
    results, samples_by_setting = {}, {}
    for system in dial_settings():
        family = system.to_dict()["family"]
        shuffle = family == "D"
        runs = load_runs(args.study / system.name / "runs")
        directions = {w: dict(np.load(args.study / "reference" / system.name / "data" / f"window_{w}.npz"))
                      for w in sorted({r["window"] for r in runs})}
        samples, excluded = build_samples(runs, directions, shuffle)
        samples_by_setting[system.name] = samples
        horizons = {f"{dt:g}": analyse(S, shuffle) for dt, S in samples.items()}
        verdict = setting_verdict(horizons[f"{PRIMARY_HORIZON:g}"], shuffle)
        results[system.name] = {"system": system.to_dict(), "family": family, "dial": system.dial,
                                "accuracy": accuracy(runs), "excluded_runs": excluded,
                                "horizons": horizons, "verdict": verdict}
        p = horizons[f"{PRIMARY_HORIZON:g}"]["N"]["M2"]
        print(f"{system.name:20s} dial {system.dial:4g}: gain {p['delta']:+.3f} [{p['ci_low']:+.3f}, "
              f"{p['ci_high']:+.3f}], coef {p['coefficient']:+.2f}, random p95 "
              f"{horizons[f'{PRIMARY_HORIZON:g}']['N']['controls']['random_p95']:+.3f} -> {verdict}", flush=True)

    families = {}
    for family in ("L", "D"):
        names = [s.name for s in dial_settings() if s.to_dict()["family"] == family]
        dials = [results[n]["dial"] for n in names]
        gains = [results[n]["horizons"][f"{PRIMARY_HORIZON:g}"]["N"]["M2"]["delta"] for n in names]
        tau = float(kendalltau(dials, gains).statistic)
        a, b = results[names[-1]]["verdict"] == "works", results[names[0]]["verdict"] == "fails"
        families[family] = {"settings": names, "dials": dials, "gains": gains, "kendall_tau": tau,
                            "largest_works": a, "smallest_fails": b, "trend_ok": tau >= KENDALL_MIN,
                            "supported": bool(a and b and tau >= KENDALL_MIN)}
    n_supported = sum(f["supported"] for f in families.values())
    overall = {2: "supported", 1: "partly supported", 0: "not supported"}[n_supported]

    lorenz = None
    if args.lorenz_runs.exists():
        runs = load_runs(args.lorenz_runs)
        windows = sorted({r["window"] for r in runs})
        directions = {w: dict(np.load(args.lorenz_lyapunov / f"lyapunov_window_{w}.npz")) for w in windows}
        samples, excluded = build_samples(runs, directions, shuffle=True)
        lorenz = {"note": "descriptive reference point; data seen before this plan", "excluded_runs": excluded,
                  "horizons": {f"{dt:g}": analyse(S, shuffle=True) for dt, S in samples.items()}}
        p = lorenz["horizons"][f"{PRIMARY_HORIZON:g}"]["N"]["M2"]
        print(f"{'lorenz63 (reference)':20s}            gain {p['delta']:+.3f} [{p['ci_low']:+.3f}, {p['ci_high']:+.3f}]")

    metrics = {"plan": "docs/dial_study_plan.md", "primary_horizon": PRIMARY_HORIZON, "settings": results,
               "families": families, "overall_verdict": overall, "lorenz_reference": lorenz}
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.out / "metrics.json")
    figures = args.out / "figures"
    dial_figure(results, lorenz, figures)
    for family in ("L", "D"):
        scatter_figure(samples_by_setting, family, figures)
    write_run_record(args.out, {"script": "experiments/26_dial_analysis.py", "study": str(args.study)},
                     REPO_ROOT, time.perf_counter() - start)
    for family, f in families.items():
        print(f"family {family}: gains {np.round(f['gains'], 3).tolist()}, Kendall tau {f['kendall_tau']:.2f}, "
              f"largest works {f['largest_works']}, smallest fails {f['smallest_fails']} -> "
              f"{'supported' if f['supported'] else 'not supported'}")
    print("overall verdict:", overall)


if __name__ == "__main__":
    main()
