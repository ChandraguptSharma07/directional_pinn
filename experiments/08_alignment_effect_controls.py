"""Exploratory (not in the analysis plan): is the fixed-magnitude alignment effect specific
to the expanding directions?

Run from the repository root (after experiments/07_directional_analysis.py):

    python experiments/08_alignment_effect_controls.py
    python experiments/08_alignment_effect_controls.py --runs results/pinn_runs_lambda10 --out results/exploratory_lambda10

The planned analysis found that, inside each run and at fixed residual size,
alignment with b1 or s1 goes with larger future error (condition 4), although
the held-out gain was small. The plan compared that effect only with zero.
This script computes the same per-run statistics (alignment effect and partial
Spearman correlation given log R) for the control directions too: random
directions and time-shuffled b1 and s1 (20 draws each), the least-expanding
directions b3 and s3, and the flow direction. If the effect is equally large
for controls, it does not reflect the local dynamics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/08_alignment_effect_controls.py` import src

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.metrics import alignment_effect, bootstrap_mean, partial_spearman
from src.analysis.plots import COLORS, apply_style, save_figure
from src.analysis.samples import N_RANDOM, N_SHUFFLE, build_samples, load_runs
from src.utils.reproducibility import load_config, save_json, write_run_record


def run_means(subset, column: str) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Mean over runs of the alignment effect and of the partial Spearman correlation for one alignment column."""
    effects, partials = [], []
    for _, group in subset.groupby("run"):
        magnitude = np.exp(group["log_R"].to_numpy())
        target = group["log_E_future"].to_numpy()
        effects.append(alignment_effect(magnitude, group[column].to_numpy(), target))
        partials.append(partial_spearman(group[column].to_numpy(), target, group["log_R"].to_numpy()))
    effects, partials = np.array(effects), np.array(partials)
    return float(np.nanmean(effects)), float(np.nanmean(partials)), effects, partials


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exploratory: alignment effect for control directions.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "results" / "pinn_runs")
    parser.add_argument("--lyapunov", type=Path, default=REPO_ROOT / "results" / "lorenz_lyapunov")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "exploratory")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_config(args.config)
    horizons = [float(x) for x in config["lorenz_lyapunov"]["horizons"]]
    runs = load_runs(args.runs)
    windows = sorted({run["summary"]["window"] for run in runs})
    lyapunov = {w: dict(np.load(args.lyapunov / "data" / f"lyapunov_window_{w}.npz")) for w in windows}
    samples = build_samples(runs, lyapunov, horizons)
    apply_style()

    results = {}
    for dt in sorted(samples["horizon"].unique()):
        subset = samples[samples["horizon"] == dt]
        entry = {}
        for name in ("b1", "s1", "b3", "s3", "flow"):
            _, _, effects, partials = run_means(subset, f"A_{name}")
            entry[name] = {"alignment_effect": bootstrap_mean(effects), "partial_spearman": bootstrap_mean(partials)}
        for family, columns in (("random", [f"A_random_{d}" for d in range(N_RANDOM)]),
                                ("shuffled_b1", [f"A_shuffled_b1_{d}" for d in range(N_SHUFFLE)]),
                                ("shuffled_s1", [f"A_shuffled_s1_{d}" for d in range(N_SHUFFLE)])):
            means = np.array([run_means(subset, c)[:2] for c in columns])
            entry[family] = {
                "alignment_effect_draws": means[:, 0].tolist(),
                "partial_spearman_draws": means[:, 1].tolist(),
                "alignment_effect_95th_percentile": float(np.quantile(means[:, 0], 0.95)),
                "partial_spearman_95th_percentile": float(np.quantile(means[:, 1], 0.95)),
            }
        results[f"{dt:g}"] = entry

    keys = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, statistic, label in ((axes[0], "alignment_effect", "fixed-size alignment effect (log units)"),
                                 (axes[1], "partial_spearman", "partial Spearman(A, $E_\\Delta$ | R)")):
        x = np.arange(len(keys))
        for name, color, offset in (("b1", COLORS["unstable"], -0.15), ("s1", COLORS["third"], 0.0),
                                    ("flow", COLORS["muted"], 0.15)):
            values = [results[k][name][statistic] for k in keys]
            mean = np.array([v["mean"] for v in values])
            ax.errorbar(x + offset, mean, yerr=[mean - [v["ci_low"] for v in values], [v["ci_high"] for v in values] - mean],
                        fmt="o", color=color, capsize=2, label={"b1": "$b_1$", "s1": "$s_1$", "flow": "flow direction"}[name])
        for family, marker in (("shuffled_b1", "x"), ("shuffled_s1", "+"), ("random", "_")):
            draws = np.array([results[k][family][f"{statistic}_draws"] for k in keys])
            ax.plot(x, np.quantile(draws, 0.95, axis=1), marker, color=COLORS["ink"], ms=9, mew=1.5, ls="none",
                    label=f"95th percentile, {family.replace('_', ' ')}")
        ax.axhline(0.0, color=COLORS["axis"], lw=1.0)
        ax.set(xticks=x, xticklabels=keys, xlabel="horizon $\\Delta t$", ylabel=label)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5)
    axes[0].set_title("(a) Alignment effect at fixed residual size (mean over runs, 95 % interval)")
    axes[1].set_title("(b) Partial rank correlation given residual size")
    save_figure(fig, args.out / "figures", "fig01_alignment_effect_controls")
    save_json({"experiment": "08_alignment_effect_controls", "runs_dir": str(args.runs), "horizons": results},
              args.out / "metrics.json")
    write_run_record(args.out, {"runs": str(args.runs)}, REPO_ROOT, time.perf_counter() - start)

    for k in keys:
        e = results[k]
        print(f"dt={k}: effect b1 {e['b1']['alignment_effect']['mean']:+.3f}, s1 {e['s1']['alignment_effect']['mean']:+.3f},"
              f" flow {e['flow']['alignment_effect']['mean']:+.3f} | 95th pct shuffled b1"
              f" {e['shuffled_b1']['alignment_effect_95th_percentile']:+.3f}, shuffled s1"
              f" {e['shuffled_s1']['alignment_effect_95th_percentile']:+.3f}, random"
              f" {e['random']['alignment_effect_95th_percentile']:+.3f}")
        print(f"        partial b1 {e['b1']['partial_spearman']['mean']:+.3f}, s1 {e['s1']['partial_spearman']['mean']:+.3f},"
              f" flow {e['flow']['partial_spearman']['mean']:+.3f} | 95th pct shuffled b1"
              f" {e['shuffled_b1']['partial_spearman_95th_percentile']:+.3f}, shuffled s1"
              f" {e['shuffled_s1']['partial_spearman_95th_percentile']:+.3f}, random"
              f" {e['random']['partial_spearman_95th_percentile']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
