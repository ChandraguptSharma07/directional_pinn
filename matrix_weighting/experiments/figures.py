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

COLORS = {"U": "#7f7f7f", "TUR": "#9467bd", "S-orc": "#ff7f0e", "M-orc": "#1f77b4", "S-self": "#ff7f0e",
          "M-self": "#1f77b4", "D-self": "#2ca02c"}
ORDER = ["U", "TUR", "S-orc", "S-self", "M-orc", "M-self"]
PLAIN = {"U": "uniform", "TUR": "Turinici", "S-orc": "scalar\n(oracle)", "S-self": "scalar\n(self)",
         "M-orc": "matrix\n(oracle)", "M-self": "matrix\n(self)"}
TITLES = {"lorenz63": "Lorenz-63 (no extra variables)", "lorenz63_damped3": "Lorenz-63 + 3 extra damped variables",
          "lorenz63_damped9": "Lorenz-63 + 9 extra damped variables", "lorenz63_damped29": "Lorenz-63 + 29 extra damped variables",
          "saddle_k1": "saddle, kappa = 1", "saddle_k4": "saddle, kappa = 4", "saddle_k16": "saddle, kappa = 16",
          "saddle_k64": "saddle, kappa = 64", "davis_skodje_g3": "Davis-Skodje, gamma = 3",
          "davis_skodje_g6": "Davis-Skodje, gamma = 6", "davis_skodje_g18": "Davis-Skodje, gamma = 18",
          "davis_skodje_g66": "Davis-Skodje, gamma = 66"}


def strip(ax, runs, keys, labels, title):
    """Per-network log10 E_rms for each method, with the median marked."""
    for i, (key, label) in enumerate(zip(keys, labels)):
        d = runs.get(key)
        if not d:
            continue
        y = np.array([log_err(v["E_rms"]) for v in d.values()])
        x = i + np.random.default_rng(i).uniform(-0.18, 0.18, size=len(y))
        ax.scatter(x, y, s=9, alpha=0.6, color=COLORS.get(key[2], "k"))
        # Black bar: the ordinary median of the errors (not the median of their logarithms).
        ax.hlines(np.log10(np.median(10.0 ** y)), i - 0.3, i + 0.3, color="k", lw=2)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("log10 of average error")
    ax.set_xlabel("method")
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
    fig, axes = plt.subplots(4, 2, figsize=(10, 13))
    for ax, s in zip(axes.ravel(), systems):
        methods = [m for m in ORDER if (s, "T2", m) in runs]
        strip(ax, runs, [(s, "T2", m) for m in methods], [PLAIN[m] for m in methods], TITLES[s])
    fig.tight_layout()
    fig.savefig(out / "gate0_errors.png", dpi=150)
    fig.savefig(out / "gate0_errors.pdf")


def gate2(out: Path):
    runs = load("gate2", lorenz_part=True)
    systems = ["lorenz63", "lorenz63_damped3", "lorenz63_damped9", "lorenz63_damped29"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, s in zip(axes.ravel(), systems):
        strip(ax, runs, [(s, "T1", m) for m in ORDER], [PLAIN[m] for m in ORDER], TITLES[s])
    fig.tight_layout()
    fig.savefig(out / "gate2_errors.png", dpi=150)
    fig.savefig(out / "gate2_errors.pdf")


def gate3(out: Path):
    runs = load("gate3")
    panels = [("rossler", "T5", "Rössler, window 5"), ("rossler", "T10", "Rössler, window 10"),
              ("lorenz96_n8", "T1", "Lorenz-96 (N=8), window 1"), ("lorenz96_n8", "T2", "Lorenz-96 (N=8), window 2")]
    methods = ["U", "TUR", "S-self", "D-self", "M-self"]
    labels = ["uniform", "Turinici", "scalar\n(no direction)", "direction\nonly", "matrix\n(Lyapunov dir.)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (system, T, title) in zip(axes.ravel(), panels):
        keys = [(system, T, m) for m in methods]
        strip(ax, runs, keys, labels, title)
        ax.set_title(title, fontsize=9, pad=16)
        for i, k in enumerate(keys):
            d = runs.get(k)
            if d:
                n_ok = sum(v["success"] for v in d.values())
                ax.annotate(f"{n_ok}/{len(d)}", (i, ax.get_ylim()[1]), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Error of each trained network (36 per method). Black bar = typical (median) network.\n"
                 "Number on top = networks whose error stayed below 1 the whole time.", fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "gate3_errors.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "gate3_errors.pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gates", nargs="+", choices=["gate0", "gate1", "gate2", "gate3"])
    args = ap.parse_args()
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for g in args.gates:
        {"gate0": gate0, "gate1": gate1, "gate2": gate2, "gate3": gate3}[g](out)
        print(f"wrote {g} figures to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
