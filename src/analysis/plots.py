"""Plain, consistent matplotlib styling shared by every figure.

Colour roles are fixed across the project so a reader learns them once:
orange marks unstable (expanding) directions and blue marks stable
(contracting) directions. The categorical colours passed a colour-vision-
deficiency check; figures still label every series, so colour is never the
only cue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "unstable": "#eb6834",  # orange
    "stable": "#2a78d6",  # blue
    "third": "#1baf7a",  # aqua, for a third kind of quantity
    "ink": "#0b0b0b",  # text and reference curves
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

# Light-to-dark steps of one blue hue, for ordered series such as prediction horizons.
ORDINAL_BLUES = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")


def apply_style() -> None:
    """Set matplotlib defaults: thin lines, light solid grid, no top/right spines."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["axis"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlesize": 10.5,
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "xtick.color": COLORS["ink_secondary"],
            "ytick.color": COLORS["ink_secondary"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,  # embed TrueType fonts so text stays editable in the PDF
        }
    )


def ordinal_colors(n: int) -> list[str]:
    """n colours from light to dark blue; n = 5 returns exactly ORDINAL_BLUES."""
    if n == 1:
        return [ORDINAL_BLUES[-1]]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("ordinal_blues", ORDINAL_BLUES)
    return [mpl.colors.to_hex(cmap(x)) for x in np.linspace(0.0, 1.0, n)]


def save_figure(fig: plt.Figure, figures_dir: str | Path, name: str) -> None:
    """Save fig as PNG (quick viewing) and PDF (vector, for papers), then close it."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)
