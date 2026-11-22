"""Render a concise presentation from the frozen independent validation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation_2026-11-23.pdf"
PREVIEWS = ROOT / "docs" / "figures"
NAVY, BLUE, TEAL, ORANGE, MUTED = "#203448", "#2878b5", "#258071", "#c66a42", "#526576"


def page(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    fig.text(0.055, 0.91, title, fontsize=25, weight="bold", color=NAVY)
    if subtitle:
        fig.text(0.057, 0.855, subtitle, fontsize=12, color=MUTED)
    fig.text(0.055, 0.035, "Lorenz-63 PINN project  |  Presentation draft  |  23 November 2026",
             fontsize=9, color="#71808d")
    return fig


def text_block(fig, x, y, heading, body, color=BLUE, width=0.27):
    fig.patches.append(plt.Rectangle((x, y - 0.25), width, 0.29, transform=fig.transFigure,
                                     facecolor="#f3f7fa", edgecolor="#d7e0e7", lw=1))
    fig.text(x + 0.018, y, heading, fontsize=15, weight="bold", color=color)
    fig.text(x + 0.018, y - 0.057, body, fontsize=12, color=NAVY, va="top", linespacing=1.5)


def main() -> None:
    quality = json.loads((ROOT / "results" / "independent_validation" / "pinn"
                          / "validation_40runs_summary.json").read_text())
    direction = json.loads((ROOT / "results" / "independent_validation" / "directional_analysis"
                            / "metrics.json").read_text())
    windows = json.loads((ROOT / "results" / "independent_validation" / "window_analysis"
                          / "metrics.json").read_text())
    if quality["n_runs"] != 40 or direction["n_runs"] != 40 or windows["n_windows"] != 8:
        raise ValueError("expected the complete frozen 40-run validation")
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    with PdfPages(OUT) as pdf:
        fig = page("Can residual direction predict future PINN error?",
                   "Original research question and hypothesis")
        text_block(fig, 0.055, 0.73, "Goal", "Improve error diagnostics\nfor chaotic Lorenz-63\nPINN trajectories.", BLUE)
        text_block(fig, 0.36, 0.73, "Hypothesis", "At equal residual size,\nalignment with an expanding\ndirection predicts more error.", TEAL)
        text_block(fig, 0.665, 0.73, "Planned test", "40 standard PINNs; held-out\ninitial-condition windows;\nrandom and shuffled controls.", ORANGE)
        fig.text(0.06, 0.29, "Original result", fontsize=16, weight="bold", color=NAVY)
        fig.text(0.06, 0.235, "Neither $b_1$ nor $s_1$ alignment met the preset decision rule.",
                 fontsize=18, color=ORANGE)
        fig.text(0.06, 0.16, "The single global PINN was accurate only over a one-unit time window.",
                 fontsize=13, color=MUTED)
        pdf.savefig(fig)
        fig.savefig(PREVIEWS / "presentation_slide_1.png", dpi=120)
        plt.close(fig)

        fig = page("A better trajectory model, tested on new windows",
                   "Eight consecutive hard-initial-condition PINNs; each new segment uses the previous predicted endpoint")
        fig.text(0.06, 0.78, f"{quality['runs_below_1_throughout']}/40", fontsize=34,
                 weight="bold", color=TEAL)
        fig.text(0.245, 0.80, "below error 1 throughout $T=2$", fontsize=16, color=NAVY)
        fig.text(0.69, 0.78, f"{quality['runs_below_1_at_least_90_percent']}/40", fontsize=34,
                 weight="bold", color=BLUE)
        fig.text(0.84, 0.80, r"for $\geq$90%" + "\nof $T=2$", fontsize=13, color=NAVY, va="top")
        ax = fig.add_axes([0.08, 0.135, 0.84, 0.54])
        ax.imshow(plt.imread(PREVIEWS / "independent_validation_40runs.png"))
        ax.axis("off")
        pdf.savefig(fig)
        fig.savefig(PREVIEWS / "presentation_slide_2.png", dpi=120)
        plt.close(fig)

        fig = page("Fresh-window prediction: one vector versus full dynamics",
                   "Leave-one-window-out prediction; exact segment joins excluded; uncertainty resamples eight whole windows")
        h = ["0.05", "0.1", "0.25", "0.5", "1"]
        x = np.arange(len(h))
        ax1 = fig.add_axes([0.08, 0.19, 0.39, 0.53])
        for name, color, offset, label in (("b1_alignment", BLUE, -0.065, "$b_1$ alignment"),
                                           ("s1_alignment", ORANGE, 0.065, "$s_1$ alignment")):
            data = [windows["horizons"][k]["models"]["residual_only"][name] for k in h]
            point = np.array([v["delta"] for v in data])
            lower = point - np.array([v["ci_low"] for v in data])
            upper = np.array([v["ci_high"] for v in data]) - point
            ax1.errorbar(x + offset, point, yerr=[lower, upper], fmt="o-", color=color,
                         capsize=3, lw=1.5, label=label)
        ax1.axhline(0, color="#777777", lw=1)
        ax1.set(xticks=x, xticklabels=h, ylim=(-0.033, 0.033), xlabel="Future horizon",
                ylabel="Gain in held-out $R^2$", title="Pointwise alignment adds no gain")
        ax1.grid(alpha=0.2)
        ax1.legend(fontsize=10, loc="lower left")
        ax2 = fig.add_axes([0.57, 0.19, 0.37, 0.53])
        data = [windows["horizons"][k]["models"]["residual_plus_time"]["oracle_amplification"] for k in h]
        point = np.array([v["delta"] for v in data])
        lower = point - np.array([v["ci_low"] for v in data])
        upper = np.array([v["ci_high"] for v in data]) - point
        ax2.errorbar(x, point, yerr=[lower, upper], fmt="o-", color=TEAL, capsize=3, lw=2)
        ax2.axhline(0, color="#777777", lw=1)
        ax2.set(xticks=x, xticklabels=h, ylim=(-0.08, 0.75), xlabel="Future horizon",
                title="Full-propagator oracle helps later")
        ax2.grid(alpha=0.2)
        fig.text(0.08, 0.105, "Oracle uses the known reference trajectory; it is not a practical predictor.",
                 fontsize=12, color=ORANGE)
        pdf.savefig(fig)
        fig.savefig(PREVIEWS / "presentation_slide_3.png", dpi=120)
        plt.close(fig)

        fig = page("Conclusion for the professor", "What the completed evidence supports")
        text_block(fig, 0.055, 0.75, "Original hypothesis", "Pointwise alignment with one\nexpanding direction was not\nsupported by held-out tests.", ORANGE)
        text_block(fig, 0.36, 0.75, "Model advance", "The sequential PINN tracks\n$T=2$ in most new runs,\nbut fails on some windows.", TEAL)
        text_block(fig, 0.665, 0.75, "Mechanistic clue", "Full dynamical amplification\nadds information, but needs\nthe true reference trajectory.", BLUE)
        fig.text(0.06, 0.32, "Next research step", fontsize=16, weight="bold", color=NAVY)
        fig.text(0.06, 0.25, "Test a causal, reference-free predictor from past residuals; ",
                 fontsize=15, color=NAVY)
        fig.text(0.06, 0.19, "validate on new windows and another dynamical system.",
                 fontsize=15, color=NAVY)
        fig.text(0.06, 0.105, "Bounded conclusion available now. The practical predictor remains open.",
                 fontsize=12, weight="bold", color=ORANGE)
        pdf.savefig(fig)
        fig.savefig(PREVIEWS / "presentation_slide_4.png", dpi=120)
        plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
