"""Create the project-report figure for the T=2 piecewise PINN pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    piece = np.load(ROOT / "results" / "piecewise_pinn_pilot" / "w0_t2_s0" / "trajectory.npz")
    global_model = np.load(ROOT / "results" / "hard_ic_pilot" / "w0_t2_s0" / "trajectory.npz")
    if not np.array_equal(piece["t"], global_model["t"]):
        raise ValueError("pilot trajectories use different time grids")
    t = piece["t"]
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    labels = ("x", "y", "z")
    for j, ax in enumerate(axes[:3]):
        ax.plot(t, piece["reference"][:, j], color="#222222", lw=1.7, label="Reference")
        ax.plot(t, global_model["pinn"][:, j], color="#c65a46", lw=1.1,
                ls="--", label="Single hard-IC PINN")
        ax.plot(t, piece["pinn"][:, j], color="#2878b5", lw=1.0,
                ls=":", label="Piecewise hard-IC PINN")
        ax.set_ylabel(labels[j])
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")
    axes[3].plot(t, np.linalg.norm(global_model["error"], axis=1), color="#c65a46",
                 lw=1.4, label="Single hard-IC PINN")
    axes[3].plot(t, np.linalg.norm(piece["error"], axis=1), color="#2878b5",
                 lw=1.4, label="Piecewise hard-IC PINN")
    axes[3].axhline(1, color="#777777", lw=0.8, ls="--")
    for boundary in np.arange(0.25, 2, 0.25):
        axes[3].axvline(boundary, color="#aaaaaa", lw=0.5, alpha=0.6)
    axes[3].set(yscale="log", ylim=(1e-5, 100), xlabel="Time", ylabel="Trajectory error norm")
    axes[3].grid(alpha=0.2)
    axes[3].legend(fontsize=8, loc="upper left")
    fig.suptitle("Lorenz-63 PINN pilot, T=2, window 0, seed 0", fontsize=12)
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "piecewise_pinn_t2_pilot.pdf")
    fig.savefig(out / "piecewise_pinn_t2_pilot.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
