"""Phase 3: tangent dynamics, Lyapunov exponents, and the two expanding directions b1 and s1.

Run from the repository root (after experiments/03_lorenz_reference.py):

    python experiments/04_lorenz_lyapunov.py

1. Lyapunov exponents from the QR method along 180 time units of the long run.
2. For every window: step propagators along its past and its reference
   trajectory; the QR method from the start of the past gives the backward
   vectors b_i(t) (spin-up of 30 time units); products of step propagators
   give Phi(t + dt, t) and, by SVD, the finite-time singular vectors
   s_i(t; dt) for every prediction horizon dt.
3. Checks: Jacobian gate, consistency with the Phase 2 reference files,
   plausible exponents, b1 independent of the starting basis, and tangent
   propagators that match nonlinear finite differences.

Outputs in results/lorenz_lyapunov/: metrics.json, figures, and
data/lyapunov_window_<k>.npz for later phases.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/04_lorenz_lyapunov.py` import src

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.plots import COLORS, apply_style, ordinal_colors, save_figure
from src.dynamics.lyapunov import (
    line_misalignment,
    lyapunov_exponents,
    make_continuous,
    qr_iteration,
    running_exponents,
    singular_directions,
)
from src.dynamics.tangent_dynamics import propagators_over, step_propagators
from src.solvers.lorenz_reference import build_windows_from_config
from src.solvers.reference_solver import integrate
from src.systems.lorenz63 import LorenzParams, lorenz_divergence, lorenz_jacobian, lorenz_rhs, verify_lorenz_jacobian
from src.utils.checks import Checks
from src.utils.reproducibility import load_config, save_json, write_run_record

COMMONLY_QUOTED = (0.9056, 0.0, -14.5723)  # e.g. Sprott, Chaos and Time-Series Analysis (2003)


def uniform_grid(start: float, stop: float, h: float) -> np.ndarray:
    return np.linspace(start, stop, int(round((stop - start) / h)) + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3: Lyapunov exponents and expanding directions.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--reference", type=Path, default=REPO_ROOT / "results" / "lorenz_reference")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "lorenz_lyapunov")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_config(args.config)
    params = LorenzParams.from_dict(config["lorenz"]["params"])
    ref_cfg, cfg = config["lorenz_reference"], config["lorenz_lyapunov"]
    if not verify_lorenz_jacobian(params).passed:  # the gate from Phase 1
        print("Jacobian check failed; not proceeding.")
        return 1
    data_dir, figures_dir = args.out / "data", args.out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    checks = Checks()
    tangent_tol = {"rtol": float(cfg["tangent_rtol"]), "atol": float(cfg["tangent_atol"])}
    length, h = float(cfg["length"]), float(cfg["dt"])
    horizons = [float(x) for x in cfg["horizons"]]
    jacobian_of = lambda states: lorenz_jacobian(states, params)

    windows = build_windows_from_config(params, ref_cfg)

    # 1. Exponents along the long run.
    t0_long = float(ref_cfg["transient"])
    t_long = uniform_grid(t0_long, t0_long + float(cfg["long_run_time"]), float(cfg["long_run_dt"]))
    long_run = windows[0].long_run
    long_props = step_propagators(lambda t: jacobian_of(long_run.at(t)), t_long, **tangent_tol)
    _, long_stretch = qr_iteration(long_props, np.eye(3))
    exponents = lyapunov_exponents(long_stretch, float(cfg["long_run_dt"]))
    running = running_exponents(long_stretch, float(cfg["long_run_dt"]))
    checks.add("long run: largest exponent within 0.1 of 0.906", abs(exponents[0] - COMMONLY_QUOTED[0]), 0.1)
    checks.add("long run: middle exponent within 0.1 of 0", abs(exponents[1]), 0.1)
    checks.add("long run: smallest exponent within 0.3 of -14.57", abs(exponents[2] - COMMONLY_QUOTED[2]), 0.3)
    checks.add("long run: sum of exponents vs trace -(sigma+1+beta)", abs(exponents.sum() - lorenz_divergence(params)),
               1e-6)

    # 2. Per window: b_i from the QR method with spin-up, s_i(t; dt) from propagator products.
    t = uniform_grid(0.0, length, h)
    per_window, kept = [], {}
    rng = np.random.default_rng(0)
    for window in windows:
        stored = np.load(args.reference / "data" / f"window_{window.index}.npz")
        n_compare = min(len(stored["t"]), len(window.reference.t))
        checks.add(f"window {window.index}: rebuilt reference equals the Phase 2 file (max difference)",
                   np.abs(stored["states"][:n_compare] - window.reference.y[:n_compare]).max(), 1e-12)
        t_past = uniform_grid(-window.spinup, 0.0, float(cfg["spinup_dt"]))
        past_props = step_propagators(lambda s: jacobian_of(window.past(s)), t_past, **tangent_tol)
        props = step_propagators(lambda s: jacobian_of(window.reference.at(s)), t, **tangent_tol)
        Q_past, _ = qr_iteration(past_props, np.eye(3))
        Q, log_stretch = qr_iteration(props, Q_past[-1])
        Q_other_past, _ = qr_iteration(past_props, np.linalg.qr(rng.normal(size=(3, 3)))[0])
        Q_other, _ = qr_iteration(props, Q_other_past[-1])
        checks.add(f"window {window.index}: b1 independent of the starting basis (max sin angle on the window)",
                   line_misalignment(Q[:, :, 0], Q_other[:, :, 0]).max(), 1e-8)

        states = window.reference.at(t)
        flow = lorenz_rhs(0.0, states, params)
        arrays = {
            "t": t,
            "states": states,
            "Q": Q,
            "log_stretch": log_stretch,
            "step_propagators": props,
            "flow_direction": flow / np.linalg.norm(flow, axis=1, keepdims=True),
        }
        summary = {"index": window.index, "finite_time_exponents": lyapunov_exponents(log_stretch, h), "horizons": {}}
        for dt in horizons:
            m = int(round(dt / h))
            sigma, V = singular_directions(propagators_over(props, m))
            arrays[f"sigma_{dt:g}"], arrays[f"V_{dt:g}"] = sigma, V
            angle = line_misalignment(Q[: len(V), :, 0], V[:, :, 0])
            summary["horizons"][f"{dt:g}"] = {
                "median_sin_angle_b1_s1": float(np.median(angle)),
                "fraction_sin_angle_above_0.5": float(np.mean(angle > 0.5)),
                "median_growth_rate_s1": float(np.median(np.log(sigma[:, 0]) / dt)),
                "median_growth_rate_b1": float(np.median(
                    np.log(np.linalg.norm(propagators_over(props, m) @ Q[: len(V), :, 0:1], axis=(1, 2))) / dt)),
                "median_rate_s3": float(np.median(np.log(sigma[:, 2]) / dt)),
            }
        # Tangent propagators against nonlinear finite differences at three points.
        worst = 0.0
        for j, dt in ((100, 1.0), (1200, 0.5), (2400, 0.25)):
            m = int(round(dt / h))
            delta = 1e-6 * rng.normal(size=3)
            ends = [integrate(lambda s, x: lorenz_rhs(s, x, params), states[j] + d, [t[j], t[j + m]],
                              rtol=1e-13, atol=1e-15).y[-1] for d in (np.zeros(3), delta)]
            linear = propagators_over(props[j : j + m], m)[0] @ delta
            worst = max(worst, np.linalg.norm(ends[1] - ends[0] - linear) / np.linalg.norm(linear))
        checks.add(f"window {window.index}: propagator vs nonlinear finite difference (max relative error)", worst, 1e-4)
        np.savez(data_dir / f"lyapunov_window_{window.index}.npz", **arrays)
        per_window.append(summary)
        if window.index == 1:
            kept = arrays

    plot_exponents(t_long - t0_long, running, exponents, figures_dir)
    plot_directions(kept, horizons, figures_dir)

    metrics = {
        "experiment": "04_lorenz_lyapunov",
        "long_run_exponents": exponents,
        "commonly_quoted_exponents": COMMONLY_QUOTED,
        "long_run_time": float(cfg["long_run_time"]),
        "windows": per_window,
        "checks": checks.records,
        "all_checks_passed": checks.all_passed,
    }
    save_json(metrics, args.out / "metrics.json")
    write_run_record(args.out, {key: config[key] for key in ("lorenz", "lorenz_reference", "lorenz_lyapunov")},
                     REPO_ROOT, time.perf_counter() - start)

    print(f"Results written to {args.out}")
    print(checks.report())
    print(f"Lyapunov exponents over {cfg['long_run_time']} time units: {np.round(exponents, 4)}"
          f" (sum {exponents.sum():.6f}, trace {lorenz_divergence(params):.6f})")
    for w in per_window:
        angles = ", ".join(f"{k}: {v['median_sin_angle_b1_s1']:.2f}" for k, v in w["horizons"].items())
        print(f"window {w['index']}: median sin(angle(b1, s1)) by horizon -> {angles}")
    return 0 if checks.all_passed else 1


def plot_exponents(t, running, exponents, figures_dir) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), constrained_layout=True)
    start = 100  # skip the first time unit, where the running averages are dominated by noise
    for k, (color, name) in enumerate(((COLORS["unstable"], "$\\lambda_1$"), (COLORS["ink"], "$\\lambda_2$"))):
        axes[0].plot(t[1 + start :], running[start:, k], color=color, label=f"{name} = {exponents[k]:.3f}")
        axes[0].axhline(COMMONLY_QUOTED[k], color=color, lw=0.8, alpha=0.5)
    axes[0].set(xlabel="averaging time", ylabel="running estimate")
    axes[0].set_title("(a) Largest and middle Lyapunov exponents (thin lines: 0.906 and 0)")
    axes[0].legend(loc="upper right")
    axes[1].plot(t[1 + start :], running[start:, 2], color=COLORS["stable"], label=f"$\\lambda_3$ = {exponents[2]:.3f}")
    axes[1].axhline(COMMONLY_QUOTED[2], color=COLORS["stable"], lw=0.8, alpha=0.5)
    axes[1].set(xlabel="averaging time", ylabel="running estimate")
    axes[1].set_title("(b) Smallest exponent (thin line: -14.57)")
    axes[1].legend(loc="upper right")
    save_figure(fig, figures_dir, "fig01_lyapunov_exponents")


def plot_directions(arrays, horizons, figures_dir) -> None:
    t, states, Q = arrays["t"], arrays["states"], arrays["Q"]
    show = t <= 3.0
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(states[show, 0], states[show, 2], color=COLORS["axis"], lw=1.0)
    b1 = make_continuous(Q[:, :, 0])
    s1 = make_continuous(arrays["V_0.5"][:, :, 0])
    for j in np.arange(0, np.count_nonzero(show), 75):
        if j >= len(s1):
            break
        x, z = states[j, 0], states[j, 2]
        ax.annotate("", xy=(x + 3 * b1[j, 0], z + 3 * b1[j, 2]), xytext=(x, z),
                    arrowprops={"arrowstyle": "-|>", "color": COLORS["unstable"], "lw": 1.2})
        ax.annotate("", xy=(x + 3 * s1[j, 0], z + 3 * s1[j, 2]), xytext=(x, z),
                    arrowprops={"arrowstyle": "-|>", "color": COLORS["third"], "lw": 1.2})
    ax.plot([], [], color=COLORS["unstable"], label="$b_1(t)$ (backward, QR)")
    ax.plot([], [], color=COLORS["third"], label="$s_1(t; 0.5)$ (finite-time singular)")
    ax.set(xlabel="x", ylabel="z")
    ax.set_title("(a) Window 1, $0 \\leq t \\leq 3$: directions projected on the x-z plane")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    ax = axes[1]
    for dt, color in zip(horizons, ordinal_colors(len(horizons))):
        V = arrays[f"V_{dt:g}"]
        ax.plot(t[: len(V)], line_misalignment(Q[: len(V), :, 0], V[:, :, 0]), color=color, lw=1.0,
                label=f"$\\Delta t$ = {dt:g}")
    ax.set(xlim=(0.0, 3.0), ylim=(0.0, 1.02), xlabel="t", ylabel="sin(angle between $b_1$ and $s_1$)")
    ax.set_title("(b) How different the two directions are")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5)

    ax = axes[2]
    for dt, color in zip(horizons, ordinal_colors(len(horizons))):
        sigma = arrays[f"sigma_{dt:g}"]
        ax.plot(t[: len(sigma)], np.log(sigma[:, 0]) / dt, color=color, lw=1.0, label=f"$\\Delta t$ = {dt:g}")
    ax.axhline(COMMONLY_QUOTED[0], color=COLORS["unstable"], lw=1.0)
    ax.text(3.0, COMMONLY_QUOTED[0], " $\\lambda_1$", va="center", fontsize=8.5, color=COLORS["ink_secondary"])
    ax.set(xlim=(0.0, 3.0), xlabel="t", ylabel="$\\log \\sigma_1(t;\\Delta t)\\,/\\,\\Delta t$")
    ax.set_title("(c) Local growth rate along $s_1$ varies strongly")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5)
    save_figure(fig, figures_dir, "fig02_expanding_directions")


if __name__ == "__main__":
    sys.exit(main())
