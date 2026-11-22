"""Plot trajectory quality of the fixed piecewise PINN across 40 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results" / "piecewise_pinn_pilot")
    parser.add_argument("--out-stem", default="piecewise_pinn_validation_40runs")
    parser.add_argument("--title", default="Piecewise hard-IC PINN: 8 windows × 5 seeds")
    args = parser.parse_args()
    source = args.source
    rows = [json.loads((source / f"w{window}_t2_s{seed}" / "metrics.json").read_text())
            for window in range(1, 9) for seed in range(5)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ["#2878b5", "#459b76", "#a974ad", "#d18b37", "#c65a46"]
    for seed, color in enumerate(colors):
        group = [r for r in rows if r["seed"] == seed]
        x = [r["window"] + (seed - 2) * 0.09 for r in group]
        axes[0].scatter(x, [r["max_error"] for r in group], color=color, s=30,
                        label=f"seed {seed}", zorder=3)
        axes[1].scatter(x, [r["fraction_error_below_1"] for r in group], color=color,
                        s=30, zorder=3)
    axes[0].axhline(1, color="#777777", ls="--", lw=1)
    axes[0].set(yscale="log", ylabel="Maximum trajectory error", xlabel="Reference window",
                title="Worst error over T=2")
    axes[1].axhline(0.9, color="#777777", ls="--", lw=1)
    axes[1].set(ylim=(0, 1.05), ylabel="Fraction of T=2 with error < 1",
                xlabel="Reference window", title="Time in the accurate regime")
    for ax in axes:
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle(args.title, fontsize=12)
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.out_stem}.pdf")
    fig.savefig(out / f"{args.out_stem}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
