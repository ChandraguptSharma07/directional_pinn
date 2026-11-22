"""Turinici-style comparison figures from the saved Gate 3 runs (no new training).

For each system and window, every method's 36 networks contribute:
  (a) trajectory error during training (median and quartile band; reference used only for this plot),
  (b) equation error (unweighted residual RMS) during training,
  (c) final-time error |x_hat(T) - x(T)| per network,
  (d) whole-window RMS error per network,
  (e) one example trajectory: initial condition 1, seed 0, the same starting network for every method,
  (f) its error over time,
  (g) the equation error |r(t)| over the window (median over networks),
  (h) the residual weight over time at the end of training (median; trace/n for matrix weights).

    python experiments/turinici_style.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHODS = [("U", "uniform", "#7f7f7f"), ("TUR", "Turinici", "#9467bd"),
           ("S-self", "scalar weight (no direction)", "#ff7f0e"), ("M-self", "matrix (Lyapunov directions)", "#1f77b4")]
CASES = [("rossler", "T10", "Rössler, window [0, 10]", 0), ("rossler", "T5", "Rössler, window [0, 5]", 0),
         ("lorenz96_n8", "T1", "Lorenz-96 (N=8), window [0, 1]", 0), ("lorenz96_n8", "T2", "Lorenz-96 (N=8), window [0, 2]", 0)]


def load(system, T, m):
    d = ROOT / "results" / "gate3" / system / T / m
    return json.loads((d / "summary.json").read_text()), np.load(d / "data.npz")


def band(ax, its, values, color, label):
    v = np.array(values, dtype=float)  # (n_logs, networks)
    med, lo, hi = np.median(v, 1), np.percentile(v, 25, 1), np.percentile(v, 75, 1)
    ax.plot(its, med, color=color, lw=1.8, label=label)
    ax.fill_between(its, lo, hi, color=color, alpha=0.15)


def case_figure(system, T, title, comp, out_dir):
    fig, axes4 = plt.subplots(4, 2, figsize=(12, 17))
    axes = axes4.reshape(2, 4)  # same panel order (a)-(h), drawn two per row
    rows = []
    for m, label, color in METHODS:
        s, d = load(system, T, m)
        h, met = s["history"], s["metrics"]
        its = np.array(h["iteration"])
        band(axes[0, 0], its, h["grid_error_rms"], color, label)
        band(axes[0, 1], its, h["residual_rms"], color, label)
        eT = np.log10(np.clip(np.array(met["E_T"], float), 1e-6, 1e3))
        er = np.log10(np.clip(np.array(met["E_rms"], float), 1e-6, 1e3))
        k = [mm for mm, _, _ in METHODS].index(m)
        for ax, vals in ((axes[0, 2], eT), (axes[0, 3], er)):
            x = k + np.random.default_rng(k).uniform(-0.18, 0.18, len(vals))
            ax.scatter(x, vals, s=10, alpha=0.6, color=color)
            ax.hlines(np.median(vals), k - 0.3, k + 0.3, color="k", lw=2)
        t, xh, xr, r = d["t_eval"], d["x_eval"], d["ref_eval"], d["r_eval"]
        if m == "U":
            axes[1, 0].plot(t, xr[0, :, comp], color="k", lw=2.2, label="reference")
        axes[1, 0].plot(t, xh[0, :, comp], color=color, lw=1.3, ls="--", label=label)
        axes[1, 1].semilogy(t, np.linalg.norm(xh[0] - xr[0], axis=-1) + 1e-12, color=color, lw=1.3, label=label)
        axes[1, 2].semilogy(t, np.median(np.linalg.norm(r, axis=-1), axis=0) + 1e-12, color=color, lw=1.3, label=label)
        if h["weight_time_profile"]:
            w = np.median(np.array(h["weight_time_profile"][-1]), axis=0)
            axes[1, 3].semilogy(np.linspace(0, float(T[1:]), len(w)), w, color=color, lw=1.5, label=label)
        succ = int(np.sum(met["success_max_below_1"]))
        rows.append({"method": label, "success": f"{succ}/{len(eT)}", "median_E_T": float(10 ** np.median(eT)),
                     "median_E_rms": float(10 ** np.median(er)),
                     "median_GPU_seconds": float(s["runtime_seconds"])})
    names = [lbl for _, lbl, _ in METHODS]
    axes[0, 0].set(yscale="log", title="(a) error of the solution during training", xlabel="training step", ylabel="average error over the window")
    axes[0, 1].set(yscale="log", title="(b) how badly the equation is broken, during training", xlabel="training step",
                   ylabel="average equation error")
    for ax, ttl in ((axes[0, 2], "(c) error at the final time, every network"), (axes[0, 3], "(d) average error over the window, every network")):
        ax.set_xticks(range(len(names)), ["uniform", "Turinici", "scalar\n(no direction)", "matrix\n(Lyapunov dir.)"], fontsize=8)
        ax.set(title=ttl, ylabel="log10 error")
        ax.grid(axis="y", alpha=0.3)
    axes[1, 0].set(title=f"(e) one example run: variable {comp + 1} over time", xlabel="time t")
    axes[1, 1].set(title="(f) the same example: error over time", xlabel="time t", ylabel="error")
    axes[1, 2].set(title="(g) equation error over time (typical network)", xlabel="time t")
    axes[1, 3].set(title="(h) weight given to each time (end of training)", xlabel="time t", ylabel="weight")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7)
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    fig.suptitle(f"{title}\n36 networks per method (12 starting points × 3 seeds), same networks and "
                 f"training budget; shaded bands = middle half of the networks", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    stem = f"turinici_style_{system}_{T}"
    fig.savefig(out_dir / f"{stem}.png", dpi=140)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    return rows


def main() -> None:
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Turinici-style comparison on the Gate 3 systems (saved runs, no new training)", "",
             "Median over 36 networks. Success = error below 1 over the whole window.", ""]
    for system, T, title, comp in CASES:
        rows = case_figure(system, T, title, comp, out_dir)
        base = rows[0]
        lines += [f"## {title}", "", "| method | success | median final-time error | × vs uniform | median RMS error "
                  "| × vs uniform | GPU s (36 nets) |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['method']} | {r['success']} | {r['median_E_T']:.3g} | "
                         f"{r['median_E_T'] / base['median_E_T']:.2g} | {r['median_E_rms']:.3g} | "
                         f"{r['median_E_rms'] / base['median_E_rms']:.2g} | {r['median_GPU_seconds']:.0f} |")
        lines += ["", f"Figure: `results/figures/turinici_style_{system}_{T}.png`", ""]
    (ROOT / "results" / "gate3" / "turinici_style.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
