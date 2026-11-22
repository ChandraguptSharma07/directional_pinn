"""Figures for the matrix-weighting study (descriptive; the decisions are in analyze.py).

    python experiments/figures.py gate0 gate1 gate2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze import load, log_err  # noqa: E402

COLORS = {"U": "#7f7f7f", "TUR": "#9467bd", "S-orc": "#ff7f0e", "M-orc": "#1f77b4", "S-self": "#ffbb78",
          "M-self": "#aec7e8"}
ORDER = ["U", "TUR", "S-orc", "S-self", "M-orc", "M-self"]


def strip(ax, runs, keys, labels, title):
    """Per-network log10 E_rms for each method, with the median marked."""
    for i, (key, label) in enumerate(zip(keys, labels)):
        d = runs.get(key)
        if not d:
            continue
        y = np.array([log_err(v["E_rms"]) for v in d.values()])
        x = i + np.random.default_rng(i).uniform(-0.18, 0.18, size=len(y))
        ax.scatter(x, y, s=9, alpha=0.6, color=COLORS.get(key[2], "k"))
        ax.hlines(np.median(y), i - 0.3, i + 0.3, color="k", lw=2)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("log10 E_rms")
    ax.grid(axis="y", alpha=0.3)


def gate1(out: Path):
    runs = load("gate1")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, T in zip(axes, ("T0.5", "T1", "T2")):
        keys = [("lorenz63", T, m) for m in ORDER]
        strip(ax, runs, keys, ORDER, f"Lorenz-63, window {T[1:]}")
    fig.tight_layout()
    fig.savefig(out / "gate1_errors.png", dpi=150)
    fig.savefig(out / "gate1_errors.pdf")


def gate0(out: Path):
    runs = load("gate0")
    systems = ["saddle_k1", "saddle_k4", "saddle_k16", "saddle_k64", "davis_skodje_g3", "davis_skodje_g6",
               "davis_skodje_g18", "davis_skodje_g66"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, s in zip(axes.ravel(), systems):
        methods = [m for m in ORDER if (s, "T2", m) in runs]
        strip(ax, runs, [(s, "T2", m) for m in methods], methods, s)
    fig.tight_layout()
    fig.savefig(out / "gate0_errors.png", dpi=150)
    fig.savefig(out / "gate0_errors.pdf")


def gate2(out: Path):
    runs = load("gate2", lorenz_part=True)
    systems = ["lorenz63", "lorenz63_damped3", "lorenz63_damped9", "lorenz63_damped29"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, s in zip(axes, systems):
        strip(ax, runs, [(s, "T1", m) for m in ORDER], ORDER, f"{s} (Lorenz-part error)")
    fig.tight_layout()
    fig.savefig(out / "gate2_errors.png", dpi=150)
    fig.savefig(out / "gate2_errors.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gates", nargs="+", choices=["gate0", "gate1", "gate2"])
    args = ap.parse_args()
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for g in args.gates:
        {"gate0": gate0, "gate1": gate1, "gate2": gate2}[g](out)
        print(f"wrote {g} figures to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
