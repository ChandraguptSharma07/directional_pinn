"""Create two report figures directly from saved interpolation and weighting metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
COLORS = {"linear": "#126f7a", "hold": "#ce7d30", "box": "#7552a2",
          "uniform": "#243b53", "turinici": "#126f7a", "turinici_code": "#b45b35"}


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def interpolation_plot() -> None:
    path = ROOT / "results" / "interpolation_rebuild" / "interpolation_box_control" / "metrics.json"
    data = json.loads(path.read_text())
    horizons = ["0.05", "0.1", "0.25", "0.5"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for method, label, marker in (("linear", "Linear interpolation", "o"),
                                  ("hold", "Sample and hold", "s"),
                                  ("box", "Knot average", "^")):
        gains = [data["horizons"][h]["models"][f"{method}_0.25"]["delta_r2"] for h in horizons]
        ax.plot(range(len(horizons)), gains, marker=marker, lw=2, ms=5,
                color=COLORS[method], label=label)
    ax.axhline(0, lw=0.9, color="#778899")
    ax.set_xticks(range(len(horizons)), horizons)
    ax.set_ylim(-0.04, 0.54)
    ax.set_ylabel(r"Held-out gain in $R^2$")
    ax.set_xlabel(r"Future-error horizon $\Delta$")
    ax.set_title("Residual history: trailing width 0.25", fontsize=11, loc="left")
    ax.grid(axis="y", color="#dbe3e8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, loc="upper right", fontsize=8, frameon=False)
    save(fig, "report_interpolation_gains")


def adaptive_plot() -> None:
    root = ROOT / "results" / "adaptive_time_weighting"
    rows = json.loads((root / "sequential_seed0_summary.json").read_text())
    rows += json.loads((root / "sequential_code_seed0_summary.json").read_text())
    lookup = {(row["window"], row["weighting"]): row["median_error"] for row in rows}
    windows = list(range(1, 9))
    if len(lookup) != 24:
        raise ValueError("expected all 8 windows for all 3 weighting methods")
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for method, label, marker in (("uniform", "Uniform", "o"),
                                  ("turinici", "Paper equation", "s"),
                                  ("turinici_code", "Author code scale", "^")):
        values = [lookup[(window, method)] for window in windows]
        ax.plot(windows, values, marker=marker, ms=5, lw=1.9,
                color=COLORS[method], label=label)
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.minorticks_off()
    ax.set_ylim(0.018, 0.29)
    ax.set_xticks(windows)
    ax.set_xlabel("Lorenz starting-condition window")
    ax.set_ylabel("Median trajectory error")
    ax.grid(axis="y", which="major", color="#dbe3e8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=8, frameon=False)
    save(fig, "report_adaptive_paired_windows")


if __name__ == "__main__":
    interpolation_plot()
    adaptive_plot()
    print("Wrote report_interpolation_gains and report_adaptive_paired_windows (PDF and PNG)")
