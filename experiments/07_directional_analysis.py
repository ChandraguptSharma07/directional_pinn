"""Phases 5-8: does the residual's direction predict future PINN error beyond its size?

Run from the repository root (after experiments/04_lorenz_lyapunov.py and 06_pinn_runs.py):

    python experiments/07_directional_analysis.py
    python experiments/07_directional_analysis.py --runs results/float64_check --out results/analysis_float64

This implements docs/analysis_plan.md. The term "inconclusive" in the plan is
made precise here, before the main runs were analysed: for a direction, a
horizon "passes" if all four conditions hold; the direction is "supported" if
at least 3 horizons pass, "inconclusive" if 1-2 horizons pass or if at least 3
horizons show a meaningful held-out improvement in the predicted direction
(conditions 1 and 2) without beating the controls (3) or the fixed-magnitude
test (4), and "not supported" otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/07_directional_analysis.py` import src

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.future_error import linearized_error
from src.analysis.metrics import (
    alignment_effect,
    bootstrap_delta_r_squared,
    bootstrap_mean,
    correlations,
    leave_one_group_out,
    ols_fit,
    partial_spearman,
    r_squared,
)
from src.analysis.plots import COLORS, apply_style, ordinal_colors, save_figure
from src.analysis.projections import decompose_residual
from src.analysis.samples import DIRECTIONS, N_RANDOM, N_SHUFFLE, build_samples, load_runs
from src.utils.reproducibility import load_config, save_csv, save_json, write_run_record

N_BOOT = 2000
MIN_DELTA_R2 = 0.02
CONTROL_DIRECTIONS = ("b3", "s3", "flow")


# --- analyses ------------------------------------------------------------------------


def per_run_correlations(samples: pd.DataFrame, target: str) -> pd.DataFrame:
    predictors = ["log_R", "log_Ru_b1", "log_Ru_s1", "log_Rperp_b1", "log_Rperp_s1", "log_amp"]
    rows = []
    for (run, dt), group in samples.groupby(["run", "horizon"]):
        if len(group) < 10:  # too few points for a per-run statistic
            continue
        row = {"run": run, "window": group["window"].iat[0], "seed": group["seed"].iat[0], "horizon": dt}
        for p in predictors:
            values = group[p] + (group["log_R"] if p == "log_amp" else 0.0)  # log_amp + log_R = log |Phi r|
            c = correlations(values.to_numpy(), group[target].to_numpy())
            row[f"pearson_{p}"], row[f"spearman_{p}"] = c["pearson"], c["spearman"]
        for name in DIRECTIONS:
            row[f"alignment_effect_{name}"] = alignment_effect(np.exp(group["log_R"].to_numpy()),
                                                               group[f"A_{name}"].to_numpy(), group[target].to_numpy())
            row[f"partial_spearman_{name}"] = partial_spearman(group[f"A_{name}"].to_numpy(),
                                                               group[target].to_numpy(), group["log_R"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def held_out(samples: pd.DataFrame, target: str, extra: dict[str, np.ndarray], seed: int = 0) -> dict:
    """Leave-one-window-out predictions of the target with log R alone and with each extra feature."""
    y = samples[target].to_numpy()
    groups = samples["window"].to_numpy()
    clusters = samples["run"].to_numpy()
    base_X = samples[["log_R"]].to_numpy()
    base = leave_one_group_out(base_X, y, groups)
    result = {"M0": {"r2": r_squared(y, base), "mae": float(np.mean(np.abs(y - base))),
                     "spearman": correlations(base, y)["spearman"]}}
    for name, feature in extra.items():
        X = np.column_stack([base_X, feature])
        prediction = leave_one_group_out(X, y, groups)
        boot = bootstrap_delta_r_squared(y, base, prediction, clusters, n_boot=N_BOOT, seed=seed)
        result[name] = {**boot, "mae": float(np.mean(np.abs(y - prediction))),
                        "spearman": correlations(prediction, y)["spearman"],
                        "coefficient": float(ols_fit(X, y)[2])}
    return result


def control_deltas(samples: pd.DataFrame, target: str, columns: list[str]) -> np.ndarray:
    """Point estimates of the held-out R^2 gain for control alignments (no bootstrap)."""
    y = samples[target].to_numpy()
    groups = samples["window"].to_numpy()
    base_X = samples[["log_R"]].to_numpy()
    base_r2 = r_squared(y, leave_one_group_out(base_X, y, groups))
    return np.array([r_squared(y, leave_one_group_out(np.column_stack([base_X, samples[c]]), y, groups)) - base_r2
                     for c in columns])


def analyse_target(samples: pd.DataFrame, horizons: list[float], target: str, with_controls: bool) -> dict:
    out = {}
    per_run = per_run_correlations(samples, target)
    for k, dt in enumerate(horizons):
        subset = samples[samples["horizon"] == dt]
        if subset["window"].nunique() < 2 or len(per_run) == 0 or dt not in set(per_run["horizon"]):
            out[f"{dt:g}"] = {"n_samples": int(len(subset)), "skipped": "fewer than two windows with data"}
            continue
        runs_dt = per_run[per_run["horizon"] == dt]
        extra = {f"M1_{name}": subset[f"A_{name}"].to_numpy() for name in DIRECTIONS + CONTROL_DIRECTIONS}
        extra.update({f"M2_{name}": subset[f"log_Ru_{name}"].to_numpy() for name in DIRECTIONS})
        extra["M3_amp"] = subset["log_amp"].to_numpy()
        entry = {
            "n_samples": int(len(subset)),
            "n_runs": int(subset["run"].nunique()),
            "held_out": held_out(subset, target, extra, seed=k),
            "correlations": {
                column: {"mean": float(runs_dt[column].mean()), "sd": float(runs_dt[column].std())}
                for column in runs_dt.columns if column.startswith(("pearson_", "spearman_"))
            },
            "alignment_effect": {name: bootstrap_mean(runs_dt[f"alignment_effect_{name}"].to_numpy(), N_BOOT, seed=k)
                                 for name in DIRECTIONS},
            "partial_spearman": {name: bootstrap_mean(runs_dt[f"partial_spearman_{name}"].to_numpy(), N_BOOT, seed=k)
                                 for name in DIRECTIONS},
            "pooled": {p: correlations(subset[p] + (subset["log_R"] if p == "log_amp" else 0.0), subset[target])
                       for p in ("log_R", "log_Ru_b1", "log_Ru_s1", "log_Rperp_b1", "log_Rperp_s1", "log_amp")},
        }
        if with_controls:
            entry["controls"] = {
                "random": control_deltas(subset, target, [f"A_random_{d}" for d in range(N_RANDOM)]).tolist(),
                **{f"shuffled_{name}": control_deltas(subset, target,
                                                      [f"A_shuffled_{name}_{d}" for d in range(N_SHUFFLE)]).tolist()
                   for name in DIRECTIONS},
            }
        out[f"{dt:g}"] = entry
    return {"horizons": out, "per_run": per_run}


def decide(primary: dict, horizons: list[float]) -> dict:
    """Apply the decision rule of docs/analysis_plan.md (horizons without data count as not passing)."""
    verdicts = {}
    for name in DIRECTIONS:
        conditions = {}
        for dt in horizons:
            entry = primary["horizons"][f"{dt:g}"]
            if "skipped" in entry:
                continue
            model = entry["held_out"][f"M1_{name}"]
            controls = entry["controls"]
            c1 = model["delta"] >= MIN_DELTA_R2 and model["ci_low"] > 0
            c2 = model["coefficient"] > 0
            c3 = (model["delta"] > np.quantile(controls["random"], 0.95)
                  and model["delta"] > np.quantile(controls[f"shuffled_{name}"], 0.95))
            effect = entry["alignment_effect"][name]
            c4 = effect["mean"] > 0 and effect["ci_low"] > 0
            conditions[f"{dt:g}"] = {"1_delta_r2": bool(c1), "2_positive_coefficient": bool(c2),
                                     "3_beats_controls": bool(c3), "4_fixed_magnitude_effect": bool(c4)}
        passing = sum(all(c.values()) for c in conditions.values())
        meaningful = sum(c["1_delta_r2"] and c["2_positive_coefficient"] for c in conditions.values())
        if passing >= 3:
            label = "supported"
        elif passing >= 1 or meaningful >= 3:
            label = "inconclusive"
        else:
            label = "not supported"
        verdicts[name] = {"conditions": conditions, "horizons_passing": passing, "label": label}
    order = ["not supported", "inconclusive", "supported"]
    overall = max((v["label"] for v in verdicts.values()), key=order.index)
    return {"directions": verdicts, "overall": overall}


def linearity_diagnostics(runs: list[dict], lyapunov: dict) -> pd.DataFrame:
    """How well the linear (Duhamel) picture reproduces each run's error, and what drives the error."""
    rows = []
    for run in runs:
        if not run["summary"]["finite"]:
            continue
        lyap = lyapunov[run["summary"]["window"]]
        t, error, residual = run["t"], run["error"], run["residual"]
        h, n = t[1] - t[0], len(t)
        props = lyap["step_propagators"][: n - 1]
        linear = linearized_error(error[0], residual, props, h)
        from_initial = linearized_error(error[0], np.zeros_like(residual), props, h)
        size = np.linalg.norm(error, axis=1)
        mismatch = np.linalg.norm(linear - error, axis=1) / size
        rows.append({
            "run": run["name"], "window": run["summary"]["window"], "seed": run["summary"]["seed"],
            "median_relative_mismatch": float(np.median(mismatch)),
            "fraction_within_10_percent": float(np.mean(mismatch < 0.1)),
            "median_error": float(np.median(size)), "max_error": float(size.max()),
            "fraction_error_below_1": float(np.mean(size < 1.0)),
            "median_share_from_initial_error": float(np.median(np.linalg.norm(from_initial, axis=1) / size)),
        })
    return pd.DataFrame(rows)


# --- figures ---------------------------------------------------------------------------


def plot_single_run(run: dict, lyap: dict, figures_dir: Path) -> None:
    t, pinn, reference, residual = run["t"], run["pinn"], run["reference"], run["residual"]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6), constrained_layout=True)
    ax = axes[0, 0]
    for k, (name, color) in enumerate((("x", COLORS["unstable"]), ("y", COLORS["stable"]), ("z", COLORS["third"]))):
        ax.plot(t, reference[:, k], color=color, lw=2.4, alpha=0.35)
        ax.plot(t, pinn[:, k], color=color, lw=1.1, label=name)
    ax.set(xlabel="t", ylabel="state")
    ax.set_title(f"(a) {run['name']}: PINN (thin) and reference (wide, faint)")
    ax.legend(loc="upper right", ncol=3)
    ax = axes[0, 1]
    ax.plot(reference[:, 0], reference[:, 2], color=COLORS["axis"], lw=3.0, label="reference")
    ax.plot(pinn[:, 0], pinn[:, 2], color=COLORS["ink"], lw=1.0, label="PINN")
    ax.plot(*reference[0, [0, 2]], "o", color=COLORS["unstable"], ms=6, label="initial condition")
    ax.set(xlabel="x", ylabel="z")
    ax.set_title("(b) The same trajectories in the x-z plane")
    ax.legend(loc="upper right")
    ax = axes[1, 0]
    n = len(t)
    R = np.linalg.norm(residual, axis=1)
    b1 = lyap["Q"][:n, :, 0]
    s1 = lyap["V_0.25"][: n - 125, :, 0]
    ax.plot(t, R, color=COLORS["ink"], lw=1.2, label="$R = \\Vert r\\Vert$")
    ax.plot(t, decompose_residual(residual, b1).R_u, color=COLORS["unstable"], lw=1.0, label="$|r \\cdot b_1|$")
    ax.plot(t[: n - 125], decompose_residual(residual[: n - 125], s1).R_u, color=COLORS["third"], lw=1.0,
            label="$|r \\cdot s_1(t; 0.25)|$")
    ax.set_yscale("log")
    ax.set(xlabel="t", ylabel="residual")
    ax.set_title("(c) Residual size and its components along the expanding directions")
    ax.legend(loc="lower right", ncol=3)
    ax = axes[1, 1]
    h = t[1] - t[0]
    linear = linearized_error(run["error"][0], residual, lyap["step_propagators"][: n - 1], h)
    ax.plot(t, np.linalg.norm(run["error"], axis=1), color=COLORS["ink"], lw=1.4, label="actual error $E(t)$")
    ax.plot(t, np.linalg.norm(linear, axis=1), color=COLORS["unstable"], lw=1.0,
            label="linear reconstruction from $e(0)$ and $r$")
    ax.set_yscale("log")
    ax.set(xlabel="t", ylabel="$\\Vert \\hat x(t) - x(t)\\Vert$")
    ax.set_title("(d) Error of this run")
    ax.legend(loc="lower right")
    save_figure(fig, figures_dir, "fig01_single_run")


def plot_scatter(samples: pd.DataFrame, dt: float, figures_dir: Path) -> None:
    subset = samples[samples["horizon"] == dt]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True, sharey=True)
    for ax, (column, label) in zip(axes, (("log_R", "$\\log R$"), ("log_Ru_b1", "$\\log |r\\cdot b_1|$"),
                                          ("log_Ru_s1", f"$\\log |r\\cdot s_1(t;{dt:g})|$"))):
        x, y = subset[column].to_numpy(), subset["log_E_future"].to_numpy()
        ax.plot(x, y, ".", ms=1.5, color=COLORS["muted"], alpha=0.35)
        edges = np.quantile(x, np.linspace(0, 1, 13))
        centers, medians = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (x >= low) & (x <= high)
            centers.append(np.median(x[inside]))
            medians.append(np.median(y[inside]))
        ax.plot(centers, medians, "o-", color=COLORS["ink"], ms=4, label="binned median")
        rho = correlations(x, y)["spearman"]
        ax.set(xlabel=label)
        ax.set_title(f"pooled Spearman {rho:.2f}")
        ax.legend(loc="lower right")
    axes[0].set_ylabel(f"$\\log E_\\Delta$, $\\Delta t$ = {dt:g}")
    save_figure(fig, figures_dir, f"fig02_scatter_horizon_{dt:g}")


def plot_alignment_at_fixed_size(samples: pd.DataFrame, horizons_shown: list[float], figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, len(horizons_shown), figsize=(4.4 * len(horizons_shown), 7.2),
                             constrained_layout=True)
    colors = {"low": COLORS["stable"], "middle": COLORS["muted"], "high": COLORS["unstable"]}
    for row, name in enumerate(DIRECTIONS):
        for col, dt in enumerate(horizons_shown):
            ax = axes[row, col]
            subset = samples[samples["horizon"] == dt]
            curves = {"low": [], "middle": [], "high": []}
            for _, group in subset.groupby("run"):
                R = group["log_R"].to_numpy()
                bins = np.clip(np.searchsorted(np.quantile(R, np.linspace(0, 1, 6)), R, side="right") - 1, 0, 4)
                a, y = group[f"A_{name}"].to_numpy(), group["log_E_future"].to_numpy()
                values = {key: [] for key in curves}
                for b in range(5):
                    inside = bins == b
                    low, high = np.quantile(a[inside], [1 / 3, 2 / 3])
                    values["low"].append(np.median(y[inside & (a <= low)]) - np.median(y[inside]))
                    values["middle"].append(np.median(y[inside & (a > low) & (a < high)]) - np.median(y[inside]))
                    values["high"].append(np.median(y[inside & (a >= high)]) - np.median(y[inside]))
                for key in curves:
                    curves[key].append(values[key])
            for key, color in colors.items():
                mean = np.mean(curves[key], axis=0)
                ax.plot(np.arange(1, 6), mean, "o-", color=color, label=f"{key} alignment")
            ax.axhline(0.0, color=COLORS["axis"], lw=1.0)
            ax.set(xticks=np.arange(1, 6), xlabel="quintile of residual size $R$ (within run)")
            label = "$b_1$" if name == "b1" else f"$s_1(t;{dt:g})$"
            ax.set_title(f"{label}, $\\Delta t$ = {dt:g}")
            if col == 0:
                ax.set_ylabel("$\\log E_\\Delta$ relative to the bin median")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7.5)
    save_figure(fig, figures_dir, "fig03_alignment_at_fixed_size")


def plot_horizon_summary(primary: dict, horizons: list[float], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    ax = axes[0]
    entries = [primary["horizons"][f"{dt:g}"] for dt in horizons]
    styles = {"M1_b1": ("$A_{b_1}$", COLORS["unstable"]), "M1_s1": ("$A_{s_1}$", COLORS["third"]),
              "M3_amp": ("oracle amplification $\\Vert\\Phi r\\Vert / R$", COLORS["ink"]),
              "M1_b3": ("$A_{b_3}$ (least expanding)", COLORS["stable"]),
              "M1_flow": ("alignment with the flow", COLORS["muted"])}
    offsets = np.linspace(-0.02, 0.02, len(styles))
    for (model, (label, color)), offset in zip(styles.items(), offsets):
        delta = np.array([e["held_out"][model]["delta"] for e in entries])
        low = np.array([e["held_out"][model]["ci_low"] for e in entries])
        high = np.array([e["held_out"][model]["ci_high"] for e in entries])
        x = np.arange(len(horizons)) + offset * 10
        ax.errorbar(x, delta, yerr=[delta - low, high - delta], fmt="o-", color=color, ms=4, capsize=2, label=label)
    random95 = [np.quantile(e["controls"]["random"], 0.95) for e in entries]
    shuffled95 = [np.quantile(e["controls"]["shuffled_s1"], 0.95) for e in entries]
    ax.plot(np.arange(len(horizons)), random95, "_", color=COLORS["ink_secondary"], ms=18, mew=2,
            label="95th percentile, random directions")
    ax.plot(np.arange(len(horizons)), shuffled95, "x", color=COLORS["ink_secondary"], ms=6,
            label="95th percentile, time-shuffled $s_1$")
    ax.axhline(0.0, color=COLORS["axis"], lw=1.0)
    ax.axhline(MIN_DELTA_R2, color=COLORS["unstable"], lw=0.8, alpha=0.6)
    ax.text(len(horizons) - 0.6, MIN_DELTA_R2, " 0.02", va="center", fontsize=8, color=COLORS["ink_secondary"])
    ax.set(xticks=np.arange(len(horizons)), xticklabels=[f"{dt:g}" for dt in horizons], xlabel="horizon $\\Delta t$",
           ylabel="held-out $R^2$ gain over $\\log R$ alone")
    ax.set_title("(a) Does direction add predictive information? (95 % intervals)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)
    ax = axes[1]
    for column, label, color in (("spearman_log_R", "$R$", COLORS["ink"]), ("spearman_log_Ru_b1", "$|r\\cdot b_1|$",
                                 COLORS["unstable"]), ("spearman_log_Ru_s1", "$|r\\cdot s_1|$", COLORS["third"]),
                                 ("spearman_log_Rperp_s1", "$R_\\perp$ ($s_1$)", COLORS["stable"])):
        mean = np.array([e["correlations"][column]["mean"] for e in entries])
        sd = np.array([e["correlations"][column]["sd"] for e in entries])
        ax.errorbar(np.arange(len(horizons)), mean, yerr=sd, fmt="o-", color=color, ms=4, capsize=2, label=label)
    ax.axhline(0.0, color=COLORS["axis"], lw=1.0)
    ax.set(xticks=np.arange(len(horizons)), xticklabels=[f"{dt:g}" for dt in horizons], xlabel="horizon $\\Delta t$",
           ylabel="Spearman correlation with $E_\\Delta$ (mean ± sd over runs)")
    ax.set_title("(b) Correlation with future error by horizon")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=7.5)
    save_figure(fig, figures_dir, "fig04_horizon_summary")


def plot_runs(per_run: pd.DataFrame, horizons: list[float], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    windows = sorted(per_run["window"].unique())
    colors = dict(zip(windows, ordinal_colors(len(windows))))
    for ax, (column, label) in zip(axes, (("spearman_log_R", "Spearman($R$, $E_\\Delta$) per run"),
                                          ("alignment_effect_s1", "alignment effect for $s_1$ per run"))):
        for k, dt in enumerate(horizons):
            values = per_run[per_run["horizon"] == dt]
            jitter = np.linspace(-0.25, 0.25, len(values))
            ax.scatter(k + jitter, values[column], s=12, c=[colors[w] for w in values["window"]])
            ax.plot([k - 0.3, k + 0.3], [values[column].mean()] * 2, color=COLORS["ink"], lw=2)
        ax.axhline(0.0, color=COLORS["axis"], lw=1.0)
        ax.set(xticks=range(len(horizons)), xticklabels=[f"{dt:g}" for dt in horizons], xlabel="horizon $\\Delta t$",
               ylabel=label)
    axes[0].set_title("(a) Every run (dots, coloured by window; bar = mean)")
    axes[1].set_title("(b) Fixed-magnitude alignment effect, every run")
    save_figure(fig, figures_dir, "fig05_all_runs")


# --- main --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phases 5-8: directional residual analysis.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "results" / "pinn_runs")
    parser.add_argument("--lyapunov", type=Path, default=REPO_ROOT / "results" / "lorenz_lyapunov")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "analysis")
    parser.add_argument("--no-controls", action="store_true", help="skip the random/shuffled controls (faster)")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_config(args.config)
    horizons = [float(x) for x in config["lorenz_lyapunov"]["horizons"]]
    data_dir, figures_dir = args.out / "data", args.out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    runs = load_runs(args.runs)
    windows = sorted({run["summary"]["window"] for run in runs})
    lyapunov = {w: dict(np.load(args.lyapunov / "data" / f"lyapunov_window_{w}.npz")) for w in windows}
    failed = [run["name"] for run in runs if not run["summary"]["finite"]]
    samples = build_samples(runs, lyapunov, horizons)
    print(f"{len(runs)} runs ({len(failed)} non-finite), {len(samples)} samples", flush=True)

    with_controls = not args.no_controls
    primary = analyse_target(samples, horizons, "log_E_future", with_controls)
    secondary = analyse_target(samples, horizons, "log_N", with_controls=False)
    linear_regime = samples[(samples["log_E_now"] < 0.0) & (samples["log_E_future"] < 0.0)]
    sensitivity_linear = analyse_target(linear_regime, horizons, "log_E_future", with_controls=False)
    linearity = linearity_diagnostics(runs, lyapunov)
    decision = decide(primary, horizons) if with_controls else None

    primary["per_run"].to_csv(data_dir / "per_run_primary.csv", index=False)
    linearity.to_csv(data_dir / "linearity.csv", index=False)
    available = [dt for dt in horizons if "skipped" not in primary["horizons"][f"{dt:g}"]]
    representative = next(run for run in runs if run["summary"]["finite"])
    plot_single_run(representative, lyapunov[representative["summary"]["window"]], figures_dir)
    shown = sorted({available[0], 0.25 if 0.25 in available else available[0], available[-1]})
    for dt in shown:
        plot_scatter(samples, dt, figures_dir)
    plot_alignment_at_fixed_size(samples, shown, figures_dir)
    if with_controls:
        plot_horizon_summary(primary, available, figures_dir)
    plot_runs(primary["per_run"], available, figures_dir)

    metrics = {
        "experiment": "07_directional_analysis",
        "runs_dir": str(args.runs.relative_to(REPO_ROOT)) if args.runs.is_relative_to(REPO_ROOT) else str(args.runs),
        "n_runs": len(runs),
        "non_finite_runs": failed,
        "n_samples": int(len(samples)),
        "decision": decision,
        "primary_target_E_future": primary["horizons"],
        "secondary_target_created_error": secondary["horizons"],
        "sensitivity_linear_regime": {**sensitivity_linear["horizons"], "n_samples": int(len(linear_regime))},
        "linearity": {
            "median_relative_mismatch": float(linearity["median_relative_mismatch"].median()),
            "median_fraction_within_10_percent": float(linearity["fraction_within_10_percent"].median()),
            "median_share_from_initial_error": float(linearity["median_share_from_initial_error"].median()),
            "runs": linearity.to_dict(orient="records"),
        },
    }
    save_json(metrics, args.out / "metrics.json")
    write_run_record(args.out, {"lorenz_lyapunov": config["lorenz_lyapunov"], "runs": str(args.runs)}, REPO_ROOT,
                     time.perf_counter() - start)

    print(f"Results written to {args.out}")
    for dt in available:
        entry = primary["horizons"][f"{dt:g}"]["held_out"]
        line = "  ".join(f"{name}: {entry[name]['delta']:+.4f} [{entry[name]['ci_low']:+.4f}, {entry[name]['ci_high']:+.4f}]"
                         for name in ("M1_b1", "M1_s1", "M3_amp"))
        print(f"dt={dt:g}: R2(M0) = {entry['M0']['r2']:.3f};  delta R2  {line}")
    if decision:
        for name, verdict in decision["directions"].items():
            print(f"{name}: {verdict['label']} ({verdict['horizons_passing']} horizons pass all conditions)")
        print(f"overall: {decision['overall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
