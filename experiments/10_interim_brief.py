"""Render the interim research brief from the saved interpolation metrics.

Run from the repository root after experiment 09:
    python experiments/10_interim_brief.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/interpolation_rebuild/interpolation_box_control/metrics.json"
OUT = ROOT / "docs/interim_research_update.pdf"
COLORS = {"baseline": "#3d536a", "box": "#65a5a0", "linear": "#df8a46"}


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    if metrics.get("n_runs") != 40 or metrics.get("status") != "exploratory_result":
        raise ValueError("expected the corrected 40-run exploratory metrics")
    horizons = ["0.05", "0.1", "0.25", "0.5"]
    base = [metrics["horizons"][h]["baseline_log_pointwise_plus_time"]["r2"] for h in horizons]
    box = [metrics["horizons"][h]["models"]["box_0.25"]["r2"] for h in horizons]
    linear = [metrics["horizons"][h]["models"]["linear_0.25"]["r2"] for h in horizons]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        fig.text(0.06, 0.93, "From residual direction to residual history", fontsize=21, weight="bold", color="#233446")
        fig.text(0.06, 0.885, "Interim research update  |  Lorenz-63 PINNs  |  21 November 2026", fontsize=10, color="#526779")
        fig.text(0.06, 0.824, "Research question", fontsize=12, weight="bold", color="#233446")
        fig.text(0.06, 0.79, "Can the residual of a trained PINN indicate its future trajectory error?", fontsize=11)

        ax = fig.add_axes([0.07, 0.30, 0.56, 0.42])
        x = np.arange(len(horizons))
        width = 0.24
        for offset, values, label, color in [(-width, base, "Pointwise + time", COLORS["baseline"]),
                                             (0, box, "Plain recent average", COLORS["box"]),
                                             (width, linear, "Interpolated recent average", COLORS["linear"])]:
            ax.bar(x + offset, values, width, color=color, label=label)
        ax.set_xticks(x, horizons)
        ax.set_xlabel("Future horizon")
        ax.set_ylabel("Held-out $R^2$ for log future error")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_title("Same 40 runs, same prediction times", fontsize=11, loc="left")

        fig.text(0.67, 0.71, "What changed?", fontsize=12, weight="bold", color="#233446")
        fig.text(0.67, 0.64, "Residual vectors sampled every 0.05 time units.\nA 0.25-unit trailing average was calculated\nwith or without linear interpolation.", fontsize=10, linespacing=1.5)
        fig.text(0.67, 0.48, "Main observation", fontsize=12, weight="bold", color="#233446")
        fig.text(0.67, 0.38, "At horizons 0.10 and 0.25,\ninterpolation adds about 0.06 and 0.05\nto held-out $R^2$ over plain averaging.", fontsize=10, linespacing=1.5)
        fig.text(0.07, 0.19, "Design: 8 initial conditions × 5 seeds; leave-one-initial-condition-out prediction. The original preset",
                 fontsize=9, color="#526779")
        fig.text(0.07, 0.16, "directional-alignment test was negative. This interpolation analysis is exploratory.", fontsize=9, color="#526779")
        fig.text(0.07, 0.10, "Source: results/interpolation_rebuild/interpolation_box_control/metrics.json", fontsize=8, color="#667788")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        fig.text(0.06, 0.93, "What the result means, and what comes next", fontsize=20, weight="bold", color="#233446")
        fig.text(0.06, 0.86, "Evidence available today", fontsize=12, weight="bold", color="#233446")
        fig.text(0.06, 0.82, "1. The preset test found no reliable held-out benefit from pointwise residual alignment\n"
                 "   with expanding directions on Lorenz-63.\n"
                 "2. Recent residual magnitude improves short-horizon error prediction in the rebuilt runs.\n"
                 "3. Linear interpolation adds a modest gain over a simple average of sampled magnitudes.",
                 fontsize=10, linespacing=1.55, va="top")

        fig.text(0.06, 0.65, "Fair comparison of the submitted report's ideas", fontsize=12, weight="bold", color="#233446")
        rows = [
            ("Pointwise residual size", "Tested baseline", "Post-training diagnostic"),
            ("Plain recent average", "Tested control", "Post-training diagnostic"),
            ("Linear interpolation + average", "Tested, exploratory", "Post-training diagnostic"),
            ("Duhamel-weighted integral", "Proposed", "Post-training diagnostic"),
            ("Gaussian-process smoothing", "Proposed", "Post-training diagnostic"),
            ("Turinici time weighting", "Proposed", "Changes PINN training"),
        ]
        ax = fig.add_axes([0.06, 0.35, 0.88, 0.26])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=["Method", "Status here", "Role"],
                         loc="upper left", cellLoc="left", colLoc="left", colWidths=[0.42, 0.23, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.55)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d7e0e7")
            if row == 0:
                cell.set_facecolor("#e8eef2")
                cell.set_text_props(weight="bold", color="#233446")

        fig.text(0.06, 0.28, "Next decisive test", fontsize=12, weight="bold", color="#233446")
        fig.text(0.06, 0.20, "Freeze the 0.25-unit comparison, test it on new untouched Lorenz windows, and measure\n"
                 "error created inside each horizon. A second dynamical system would test generality.",
                 fontsize=10, linespacing=1.5)
        fig.text(0.06, 0.10, "Limits: 8 windows; post-result controls; one system. At horizon 0.5, gain over baseline is uncertain.",
                 fontsize=8.5, color="#667788")
        pdf.savefig(fig)
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
