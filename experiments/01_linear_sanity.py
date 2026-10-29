"""Phase 0: equal-size perturbations of the linear saddle dx/dt = x, dy/dt = -y.

Run from the repository root:

    python experiments/01_linear_sanity.py
    python experiments/01_linear_sanity.py --config config/default.yaml --out results/linear_sanity

Each part produces one figure; docs/phase0_linear_saddle.md explains them.

1. Reference trajectory: exact solution vs numerical solution.
2. Two state perturbations of equal size, along the unstable and the stable direction.
3. Two approximate solutions whose residuals have equal size at every time,
   one along the unstable and one along the stable direction.
4. Alignment sweep: error after a horizon dt as a function of how well a
   residual (or perturbation) of fixed size aligns with the unstable direction,
   with an isotropic system as a negative control.
5. Sheared (non-normal) saddle: different definitions of "the unstable
   direction" disagree, and only some of them predict growth.

Every numerical result is compared with its closed form. The comparisons are
saved in metrics.json under "checks", and the script exits with status 1 if
any of them fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/01_linear_sanity.py` import src

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.plots import COLORS, apply_style, ordinal_colors, save_figure
from src.analysis.projections import decompose_residual
from src.solvers.reference_solver import integrate
from src.systems.linear_saddle import (
    amplification,
    direction_angle,
    isotropic,
    leading_singular_direction,
    lyapunov_directions_2d,
    saddle,
    saddle_exact_solution,
    saddle_forced_error,
    sheared_saddle,
    unit_vectors,
)
from src.utils.checks import Checks
from src.utils.reproducibility import load_config, save_csv, save_json, write_run_record

UNSTABLE = np.array([1.0, 0.0])  # x axis: perturbations grow like e^t
STABLE = np.array([0.0, 1.0])  # y axis: perturbations shrink like e^-t
CASES = {"unstable": UNSTABLE, "stable": STABLE}
CASE_COLORS = {"unstable": COLORS["unstable"], "stable": COLORS["stable"]}


def relative_error(approx: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Row-wise |approx - exact| / |exact| (Euclidean norms)."""
    return np.linalg.norm(approx - exact, axis=-1) / np.linalg.norm(exact, axis=-1)


def line_misalignment(u: np.ndarray, v: np.ndarray) -> float:
    """|u_x v_y - u_y v_x|: sine of the angle between the lines spanned by unit vectors u and v."""
    return float(abs(u[0] * v[1] - u[1] * v[0]))


def with_constant_residual(rhs, r: np.ndarray):
    """Right-hand side of dz/dt = f(t, z) + r. Every solution of it has residual exactly r."""
    return lambda t, z: rhs(t, z) + r


def marker_indices(n: int, n_markers: int = 20) -> np.ndarray:
    """About n_markers evenly spaced indices into an array of length n."""
    return np.arange(0, n, max(1, n // n_markers))


def note(ax, text: str, x: float = 0.02, y: float = 0.03) -> None:
    """Small grey annotation at axes coordinates (x, y)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, color=COLORS["ink_secondary"])


# --- Part 1 -----------------------------------------------------------------------


def part1_reference(cfg, t, solver, checks, data_dir, figures_dir):
    """Exact and numerical solution of the saddle from the reference initial condition."""
    system = saddle()
    z0 = np.asarray(cfg["z0"], dtype=float)
    exact = saddle_exact_solution(t, z0)
    numerical = integrate(system.rhs, z0, t, **solver)
    rel_err = relative_error(numerical.y, exact)
    checks.add("part1: numerical vs exact trajectory, max relative error", rel_err.max(), 1e-8)
    checks.add(
        "part1: matrix-exponential solution vs closed form, max relative error",
        relative_error(system.solution(t, z0), exact).max(),
        1e-12,
    )
    save_csv(
        data_dir / "fig01_reference.csv",
        {
            "t": t,
            "x_exact": exact[:, 0],
            "y_exact": exact[:, 1],
            "x_numerical": numerical.y[:, 0],
            "y_numerical": numerical.y[:, 1],
            "relative_error": rel_err,
        },
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    ax = axes[0]
    X, Y = np.meshgrid(np.linspace(-1.6, 1.6, 33), np.linspace(-1.2, 1.2, 25))
    ax.streamplot(X, Y, X, -Y, color=COLORS["axis"], linewidth=0.7, density=0.8, arrowsize=0.8)
    ax.axhline(0.0, color=COLORS["unstable"], lw=2.2, label="unstable direction (x axis)")
    ax.axvline(0.0, color=COLORS["stable"], lw=2.2, label="stable direction (y axis)")
    ax.plot(exact[:, 0], exact[:, 1], color=COLORS["ink"], lw=1.8, label=f"reference trajectory, $0 \\leq t \\leq {t[-1]:g}$")
    ax.plot(*z0, "o", color=COLORS["ink"], ms=5)
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.2, 1.2), xlabel="x", ylabel="y")
    ax.set_title("(a) Phase portrait: $\\dot x = x,\\ \\dot y = -y$")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="none")

    ax = axes[1]
    idx = marker_indices(len(t))
    for k, (name, color) in enumerate((("$x(t)$", COLORS["unstable"]), ("$y(t)$", COLORS["stable"]))):
        ax.plot(t, exact[:, k], color=color, label=name)
        ax.plot(t[idx], numerical.y[idx, k], "o", mfc="none", mec=color, ms=5)
    ax.set_yscale("log")
    ax.set(xlabel="t", ylabel="state component (log scale)")
    ax.set_title("(b) $x = x_0 e^{t}$ grows, $y = y_0 e^{-t}$ decays")
    ax.legend(loc="center right")
    note(ax, "lines: exact formula    circles: numerical (DOP853)")

    ax = axes[2]
    positive = rel_err > 0
    ax.plot(t[positive], rel_err[positive], color=COLORS["ink"], lw=1.2, label="relative error")
    ax.axhline(solver["rtol"], color=COLORS["muted"], lw=1.0, label=f"solver rtol = {solver['rtol']:g}")
    ax.set_yscale("log")
    ax.set(xlabel="t", ylabel="$\\Vert z_{num} - z_{exact}\\Vert \\,/\\, \\Vert z_{exact}\\Vert$")
    ax.set_title("(c) Numerical solution vs exact solution")
    ax.legend(loc="lower right")
    save_figure(fig, figures_dir, "fig01_saddle_reference")

    return numerical, {"max_relative_error": rel_err.max(), "solver_stats": numerical.stats}


# --- Part 2 -----------------------------------------------------------------------


def part2_state_perturbations(cfg, t, reference, solver, checks, data_dir, figures_dir):
    """Perturb the initial state by eps along each direction and follow the difference."""
    system = saddle()
    z0 = np.asarray(cfg["z0"], dtype=float)
    eps = float(cfg["epsilon"])
    columns = {"t": t}
    sizes, sizes_exact = {}, {}
    for case, direction in CASES.items():
        delta0 = eps * direction
        perturbed = integrate(system.rhs, z0 + delta0, t, **solver)
        delta_numerical = perturbed.y - reference.y
        delta_exact = saddle_exact_solution(t, delta0)  # a perturbation obeys the same linear ODE
        # Tolerance 1e-3: the stable perturbation shrinks to ~1e-5 of the state size, so the
        # subtraction perturbed - reference amplifies the solver's relative error by ~1e5.
        checks.add(
            f"part2: {case} perturbation, numerical vs exact, max relative error",
            relative_error(delta_numerical, delta_exact).max(),
            1e-3,
        )
        sizes[case] = np.linalg.norm(delta_numerical, axis=1)
        sizes_exact[case] = np.linalg.norm(delta_exact, axis=1)
        columns[f"size_{case}_numerical"] = sizes[case]
        columns[f"size_{case}_exact"] = sizes_exact[case]
    ratio = sizes["unstable"] / sizes["stable"]
    checks.add(
        "part2: size ratio unstable/stable vs exp(2t), max relative error",
        np.max(np.abs(ratio / np.exp(2 * t) - 1)),
        1e-3,
    )
    columns["ratio_numerical"] = ratio
    columns["ratio_exact"] = np.exp(2 * t)
    save_csv(data_dir / "fig02_state_perturbations.csv", columns)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    idx = marker_indices(len(t))
    ax = axes[0]
    for case in CASES:
        ax.plot(t, sizes_exact[case], color=CASE_COLORS[case], label=f"perturbation along {case} direction")
        ax.plot(t[idx], sizes[case][idx], "o", mfc="none", mec=CASE_COLORS[case], ms=5)
    ax.axhline(eps, color=COLORS["muted"], lw=1.0)
    ax.text(t[-1], eps * 1.3, f"initial size $\\varepsilon$ = {eps:g}", ha="right", va="bottom", fontsize=8,
            color=COLORS["ink_secondary"])
    ax.set_yscale("log")
    ax.set(xlabel="t", ylabel="perturbation size $\\Vert\\delta(t)\\Vert$")
    ax.set_title("(a) Same initial size, opposite fates")
    ax.legend(loc="upper left")
    note(ax, "lines: $\\varepsilon e^{t}$ and $\\varepsilon e^{-t}$    circles: numerical")

    ax = axes[1]
    ax.plot(t, np.exp(2 * t), color=COLORS["ink"], label="exact: $e^{2t}$")
    ax.plot(t[idx], ratio[idx], "o", mfc="none", mec=COLORS["ink"], ms=5, label="numerical")
    ax.set_yscale("log")
    ax.set(xlabel="t", ylabel="size ratio, unstable / stable")
    ax.set_title("(b) Ratio of the two perturbation sizes")
    ax.legend(loc="upper left")
    save_figure(fig, figures_dir, "fig02_state_perturbations")

    T = float(t[-1])
    return {
        "epsilon": eps,
        "final_size": {case: sizes[case][-1] for case in CASES},
        "final_size_exact": {"unstable": eps * np.exp(T), "stable": eps * np.exp(-T)},
        "final_ratio": ratio[-1],
        "final_ratio_exact": np.exp(2 * T),
    }


# --- Part 3 -----------------------------------------------------------------------


def part3_constant_residuals(cfg, t, reference, solver, checks, data_dir, figures_dir):
    """Two approximate solutions with residuals of equal size eps but different direction."""
    system = saddle()
    z0 = np.asarray(cfg["z0"], dtype=float)
    eps = float(cfg["epsilon"])
    columns = {"t": t}
    out = {}
    for case, direction in CASES.items():
        r = eps * direction
        # An approximate solution that starts exactly at z0 but violates the ODE by r at every time.
        approx = integrate(with_constant_residual(system.rhs, r), z0, t, **solver)
        error_numerical = approx.y - reference.y
        error_exact = saddle_forced_error(t, r)
        checks.add(
            f"part3: {case} residual, error numerical vs exact (t > 0), max relative error",
            relative_error(error_numerical[1:], error_exact[1:]).max(),
            1e-3,
        )
        # Measure the residual the way a PINN would: d(z_hat)/dt - f(z_hat).
        residual = system.residual(t, approx.y)
        checks.add(
            f"part3: {case} residual, measured vs prescribed, max |difference| / eps",
            np.linalg.norm(residual - r, axis=1).max() / eps,
            5e-3,
        )
        parts = decompose_residual(residual, UNSTABLE)
        out[case] = {
            "R": parts.R,
            "R_u": parts.R_u,
            "R_perp": parts.R_perp,
            "alignment": parts.alignment,
            "E_numerical": np.linalg.norm(error_numerical, axis=1),
            "E_exact": np.linalg.norm(error_exact, axis=1),
        }
        columns.update({f"{key}_{case}": value for key, value in out[case].items()})
    save_csv(data_dir / "fig03_constant_residuals.csv", columns)

    labels = {"unstable": "residual along unstable direction", "stable": "residual along stable direction"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    ax = axes[0]
    ax.plot(t, out["unstable"]["R"] / eps, color=CASE_COLORS["unstable"], lw=4.0, label=labels["unstable"])
    ax.plot(t, out["stable"]["R"] / eps, color=CASE_COLORS["stable"], lw=1.5, label=labels["stable"])
    ax.set(ylim=(0.0, 1.4), xlabel="t", ylabel="$R(t) \\,/\\, \\varepsilon$")
    ax.set_title("(a) Residual size $R = \\Vert r\\Vert$: identical")
    ax.legend(loc="lower right")

    ax = axes[1]
    for case in CASES:
        ax.plot(t, out[case]["R_u"] / eps, color=CASE_COLORS[case], label=labels[case])
    ax.set(ylim=(-0.05, 1.4), xlabel="t", ylabel="$R_u(t) \\,/\\, \\varepsilon$")
    ax.set_title("(b) Residual along unstable direction $R_u = |r \\cdot u_1|$")
    ax.legend(loc="center right")

    ax = axes[2]
    idx = marker_indices(len(t))[1:]  # skip t = 0, where the error is exactly 0
    for case in CASES:
        ax.plot(t[1:], out[case]["E_exact"][1:], color=CASE_COLORS[case], label=labels[case])
        ax.plot(t[idx], out[case]["E_numerical"][idx], "o", mfc="none", mec=CASE_COLORS[case], ms=5)
    ax.set_yscale("log")
    ax.set(xlabel="t", ylabel="$E(t) = \\Vert \\hat z(t) - z(t)\\Vert$")
    ax.set_title("(c) True error: very different")
    ax.legend(loc="lower right")
    note(ax, "lines: $\\varepsilon(e^{t}-1)$, $\\varepsilon(1-e^{-t})$    circles: numerical", y=0.93)
    save_figure(fig, figures_dir, "fig03_constant_residuals")

    horizons = [float(h) for h in cfg["horizons"]]
    error_at = {case: np.interp(horizons, t, out[case]["E_numerical"]) for case in CASES}
    return {
        "residual_over_eps": {
            case: {
                key: {"min": (out[case][key] / eps).min(), "max": (out[case][key] / eps).max()}
                for key in ("R", "R_u", "R_perp")
            }
            for case in CASES
        },
        "error_after_horizon_from_t0": [
            {
                "horizon": h,
                "unstable": error_at["unstable"][i],
                "stable": error_at["stable"][i],
                "ratio": error_at["unstable"][i] / error_at["stable"][i],
                "ratio_exact": np.exp(h),
            }
            for i, h in enumerate(horizons)
        ],
        "final_error": {case: out[case]["E_numerical"][-1] for case in CASES},
        "final_ratio": out["unstable"]["E_numerical"][-1] / out["stable"]["E_numerical"][-1],
        "final_ratio_exact": np.exp(t[-1]),
    }


# --- Part 4 -----------------------------------------------------------------------


def spot_check(system, dt: float, solver: dict) -> float:
    """Largest relative difference between integrated and closed-form results for three directions."""
    worst = 0.0
    for v in unit_vectors(np.radians([0.0, 45.0, 90.0])):
        state = integrate(system.rhs, v, [0.0, dt], **solver).y[-1]
        forced = integrate(with_constant_residual(system.rhs, v), np.zeros(2), [0.0, dt], **solver).y[-1]
        worst = max(
            worst,
            np.linalg.norm(state - system.propagator(dt) @ v) / np.linalg.norm(state),
            np.linalg.norm(forced - system.forced_error_operator(dt) @ v) / np.linalg.norm(forced),
        )
    return worst


def part4_alignment_sweep(cfg, solver, checks, data_dir, figures_dir):
    """Error after a horizon dt for every direction of a fixed-size perturbation or residual."""
    angles = np.linspace(0.0, np.pi, int(cfg["n_angles"]), endpoint=False)
    directions = unit_vectors(angles)
    alignment = decompose_residual(directions, UNSTABLE).alignment  # = |cos(angle)|
    horizons = [float(h) for h in cfg["horizons"]]
    systems = {"saddle": saddle(), "isotropic": isotropic(float(cfg["isotropic_rate"]))}

    sweep = {}
    long_table = {"system": [], "horizon": [], "angle_deg": [], "alignment": [], "state_growth": [],
                  "residual_error_per_dt": []}
    for name, system in systems.items():
        for dt in horizons:
            # |delta(t + dt)| / |delta(t)| for a unit state perturbation
            state_growth = amplification(system.propagator(dt), directions)
            # |e(t + dt)| / (|r| dt) for a constant unit residual on [t, t + dt], starting from zero error
            residual_error = amplification(system.forced_error_operator(dt), directions) / dt
            sweep[name, dt] = (state_growth, residual_error)
            checks.add(f"part4: {name}, dt={dt:g}, closed forms vs numerical integration, max relative error",
                       spot_check(system, dt, solver), 1e-8)
            long_table["system"] += [name] * len(angles)
            long_table["horizon"] += [dt] * len(angles)
            long_table["angle_deg"] += list(np.degrees(angles))
            long_table["alignment"] += list(alignment)
            long_table["state_growth"] += list(state_growth)
            long_table["residual_error_per_dt"] += list(residual_error)
    save_csv(data_dir / "fig04_alignment_sweep.csv", long_table)

    table = []
    for dt in horizons:
        state, residual = sweep["saddle", dt]
        iso_state, iso_residual = sweep["isotropic", dt]
        row = {
            "horizon": dt,
            "state_ratio": state.max() / state.min(),
            "state_ratio_exact": np.exp(2 * dt),
            "residual_ratio": residual.max() / residual.min(),
            "residual_ratio_exact": np.exp(dt),
            "isotropic_state_ratio": iso_state.max() / iso_state.min(),
            "isotropic_residual_ratio": iso_residual.max() / iso_residual.min(),
        }
        checks.add(f"part4: saddle, dt={dt:g}, state ratio vs exp(2 dt), relative error",
                   abs(row["state_ratio"] / row["state_ratio_exact"] - 1), 1e-12)
        checks.add(f"part4: saddle, dt={dt:g}, residual ratio vs exp(dt), relative error",
                   abs(row["residual_ratio"] / row["residual_ratio_exact"] - 1), 1e-12)
        checks.add(f"part4: isotropic, dt={dt:g}, max/min ratio - 1",
                   max(row["isotropic_state_ratio"], row["isotropic_residual_ratio"]) - 1, 1e-12)
        table.append(row)
    save_csv(data_dir / "fig04_ratio_vs_horizon.csv", {key: [row[key] for row in table] for key in table[0]})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    first_quadrant = angles <= np.pi / 2 + 1e-12  # alignment runs once from 1 to 0 on [0°, 90°]
    order = np.argsort(alignment[first_quadrant])
    for dt, color in zip(horizons, ordinal_colors(len(horizons))):
        state, residual = sweep["saddle", dt]
        label = f"$\\Delta t$ = {dt:g}"
        axes[0].plot(alignment[first_quadrant][order], state[first_quadrant][order], color=color, label=label)
        axes[1].plot(alignment[first_quadrant][order], residual[first_quadrant][order], color=color, label=label)
    axes[0].set(xlabel="alignment $A = |v \\cdot u_1|$", ylabel="$\\Vert\\delta(t+\\Delta t)\\Vert \\,/\\, \\Vert\\delta(t)\\Vert$")
    axes[0].set_title("(a) State perturbation of fixed size")
    axes[1].set(xlabel="alignment $A = |r \\cdot u_1| \\,/\\, \\Vert r\\Vert$",
                ylabel="$\\Vert e(t+\\Delta t)\\Vert \\,/\\, (\\Vert r\\Vert \\Delta t)$")
    axes[1].set_title("(b) Constant residual of fixed size on $[t, t+\\Delta t]$")
    for ax in axes[:2]:
        ax.legend(loc="upper left")

    ax = axes[2]
    fine = np.linspace(0.0, max(horizons), 200)
    ax.plot(fine, np.exp(2 * fine), color=COLORS["ink"], label="state perturbation: $e^{2\\Delta t}$")
    ax.plot(fine, np.exp(fine), color=COLORS["third"], label="constant residual: $e^{\\Delta t}$")
    ax.plot(fine, np.ones_like(fine), color=COLORS["muted"], label="isotropic control (both): 1")
    for key, color in (("state_ratio", COLORS["ink"]), ("residual_ratio", COLORS["third"])):
        ax.plot(horizons, [row[key] for row in table], "o", mfc="none", mec=color, ms=6)
    ax.plot(horizons, [row["isotropic_state_ratio"] for row in table], "o", mfc="none", mec=COLORS["muted"], ms=6)
    ax.set(xlabel="horizon $\\Delta t$", ylabel="error ratio: aligned ($A=1$) / perpendicular ($A=0$)")
    ax.set_title("(c) How much orientation can matter")
    ax.legend(loc="upper left")
    note(ax, "lines: closed form    circles: alignment sweep", y=0.62)
    save_figure(fig, figures_dir, "fig04_alignment_vs_future_error")

    return {"ratio_table": table}


# --- Part 5 -----------------------------------------------------------------------


def part5_sheared_saddle(cfg, checks, data_dir, figures_dir):
    """Non-normal saddle: which notion of 'unstable direction' predicts growth?"""
    c = float(cfg["shear"])
    system = sheared_saddle(c)
    named = lyapunov_directions_2d(system.A)

    # Closed forms (derived in docs/phase0_linear_saddle.md).
    n = np.hypot(2.0, c)
    closed_form = {"clv1": [1.0, 0.0], "b2": [0.0, 1.0], "clv2": [-c / n, 2.0 / n], "f1": [2.0 / n, c / n]}
    for key, v in closed_form.items():
        checks.add(f"part5: {key} vs closed form, sin(angle)", line_misalignment(named[key], np.array(v)), 1e-12)

    # Independent numerical routes: repeated QR gives the backward vector b1,
    # the SVD of a long-time propagator gives the forward vectors f1 and f2.
    angle0 = 0.5  # generic orthonormal starting basis
    q = np.array([[np.cos(angle0), -np.sin(angle0)], [np.sin(angle0), np.cos(angle0)]])
    step = system.propagator(0.5)
    for _ in range(40):  # total time 20
        q, _ = np.linalg.qr(step @ q)
    _, _, vt = np.linalg.svd(system.propagator(20.0))
    checks.add("part5: b1 from repeated QR vs clv1, sin(angle)", line_misalignment(q[:, 0], named["b1"]), 1e-10)
    checks.add("part5: f1 from SVD of Phi(20) vs closed form, sin(angle)", line_misalignment(vt[0], named["f1"]), 1e-10)
    checks.add("part5: f2 from SVD of Phi(20) vs clv2, sin(angle)", line_misalignment(vt[1], named["f2"]), 1e-10)

    angles = np.linspace(0.0, np.pi, int(cfg["n_angles"]), endpoint=False)
    directions = unit_vectors(angles)
    horizons = [float(h) for h in cfg["shear_horizons"]]
    w = np.array([1.0, c / 2.0])  # left eigenvector for eigenvalue +1, scaled so that w . clv1 = 1
    keys = ("b1", "b2", "f1", "f2")
    curves, per_horizon = {}, []
    long_table = {"horizon": [], "angle_deg": [], "growth": [], "alignment_b1": [], "alignment_s1": []}
    for dt in horizons:
        phi = system.propagator(dt)
        growth = amplification(phi, directions)
        sigma = np.linalg.svd(phi, compute_uv=False)
        _, s1 = leading_singular_direction(phi)
        along_s1 = np.abs(directions @ s1)
        predicted = np.sqrt(sigma[1] ** 2 + (sigma[0] ** 2 - sigma[1] ** 2) * along_s1**2)
        checks.add(f"part5: dt={dt:g}, growth vs singular-vector formula, max relative error",
                   np.max(np.abs(growth / predicted - 1)), 1e-10)
        curves[dt] = (growth, s1)
        long_table["horizon"] += [dt] * len(angles)
        long_table["angle_deg"] += list(np.degrees(angles))
        long_table["growth"] += list(growth)
        long_table["alignment_b1"] += list(np.abs(directions @ named["b1"]))
        long_table["alignment_s1"] += list(along_s1)
        per_horizon.append(
            {
                "horizon": dt,
                "most_amplified_direction_deg": direction_angle(s1),
                "max_growth": sigma[0],
                "growth": {key: amplification(phi, named[key][None, :])[0] for key in keys},
                "growth_over_exp_dt": {key: amplification(phi, named[key][None, :])[0] / np.exp(dt) for key in keys},
            }
        )
    save_csv(data_dir / "fig05_sheared_growth.csv", long_table)

    symmetric_part = 0.5 * (system.A + system.A.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_part)
    instantaneous = eigenvectors[:, np.argmax(eigenvalues)]

    styles = {
        "b1": ("$b_1$ = CLV$_1$ (unstable eigenvector)", COLORS["unstable"]),
        "f2": ("$f_2$ = CLV$_2$ (stable eigenvector)", COLORS["stable"]),
        "f1": ("$f_1$ (perpendicular to stable eigenvector)", COLORS["third"]),
        "b2": ("$b_2$ (perpendicular to unstable eigenvector)", COLORS["muted"]),
    }
    short = {"b1": "$b_1$=CLV$_1$", "f1": "$f_1$", "b2": "$b_2$", "f2": "CLV$_2$=$f_2$"}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)

    ax = axes[0]
    X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 31), np.linspace(-1.5, 1.5, 31))
    ax.streamplot(X, Y, X + c * Y, -Y, color=COLORS["axis"], linewidth=0.7, density=0.8, arrowsize=0.8)
    for key, (label, color) in styles.items():
        v = 1.45 * named[key]
        ax.plot([-v[0], v[0]], [-v[1], v[1]], color=color, lw=2.2, label=f"{label}: {direction_angle(named[key]):.0f}°")
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel="x", ylabel="y")
    ax.set_title(f"(a) Sheared saddle $\\dot x = x + {c:g}y,\\ \\dot y = -y$")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)

    ax = axes[1]
    degrees = np.degrees(angles)
    for dt, color in zip(horizons, ordinal_colors(len(horizons))):
        growth, _ = curves[dt]
        k = int(np.argmax(growth))
        ax.plot(degrees, growth, color=color, label=f"$\\Delta t$ = {dt:g} (max at {degrees[k]:.1f}°)")
        ax.plot(degrees[k], growth[k], "o", color=color, ms=6)
    for key in keys:
        angle = direction_angle(named[key])
        ax.axvline(angle, color=COLORS["muted"], lw=0.9)
        ax.text(angle, 1.01, short[key], transform=ax.get_xaxis_transform(), ha="left" if angle == 0 else "center",
                va="bottom", fontsize=8.5, color=COLORS["ink_secondary"])
    ax.set_yscale("log")
    ax.set(xlim=(0.0, 180.0), xticks=np.arange(0, 181, 45), xlabel="direction of unit perturbation $v$ (degrees)",
           ylabel="growth $\\Vert\\Phi(\\Delta t)\\, v\\Vert$")
    ax.set_title("(b) Growth by direction", pad=16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)

    ax = axes[2]
    dt_c = 1.0 if 1.0 in horizons else horizons[len(horizons) // 2]
    growth, s1 = curves[dt_c]
    along_b1 = np.abs(directions @ named["b1"])
    along_s1 = np.abs(directions @ s1)
    order = np.argsort(along_s1)
    ax.plot(along_b1, growth, "o", ms=3, color=COLORS["unstable"], label="vs alignment with $b_1$ (unstable eigenvector)")
    ax.plot(along_s1[order], growth[order], color=COLORS["third"], lw=2.2,
            label=f"vs alignment with $s_1$ (most-amplified direction, {direction_angle(s1):.1f}°)")
    ax.set(xlabel="alignment $|v \\cdot u|$", ylabel="growth $\\Vert\\Phi(\\Delta t)\\, v\\Vert$")
    ax.set_title(f"(c) Same perturbations, two alignment measures ($\\Delta t$ = {dt_c:g})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    save_figure(fig, figures_dir, "fig05_sheared_saddle_directions")

    return {
        "shear": c,
        "direction_angles_deg": {key: direction_angle(v) for key, v in named.items()},
        "instantaneous_most_amplified_direction_deg": direction_angle(instantaneous),
        "asymptotic_growth_coefficient": {key: abs(w @ named[key]) for key in keys},
        "per_horizon": per_horizon,
    }


# --- main -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0: equal-size perturbations of a linear saddle.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "linear_sanity")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    cfg = load_config(args.config)["linear_saddle"]
    solver = {
        "method": str(cfg["solver"]["method"]),
        "rtol": float(cfg["solver"]["rtol"]),
        "atol": float(cfg["solver"]["atol"]),
    }
    data_dir, figures_dir = args.out / "data", args.out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    t = np.linspace(0.0, float(cfg["t_final"]), int(cfg["n_time"]))
    checks = Checks()
    reference, part1 = part1_reference(cfg, t, solver, checks, data_dir, figures_dir)
    metrics = {
        "experiment": "01_linear_sanity",
        "solver": solver,
        "part1_reference": part1,
        "part2_state_perturbations": part2_state_perturbations(cfg, t, reference, solver, checks, data_dir, figures_dir),
        "part3_constant_residuals": part3_constant_residuals(cfg, t, reference, solver, checks, data_dir, figures_dir),
        "part4_alignment_sweep": part4_alignment_sweep(cfg, solver, checks, data_dir, figures_dir),
        "part5_sheared_saddle": part5_sheared_saddle(cfg, checks, data_dir, figures_dir),
        "checks": checks.records,
        "all_checks_passed": checks.all_passed,
    }
    save_json(metrics, args.out / "metrics.json")
    write_run_record(args.out, {"linear_saddle": cfg}, REPO_ROOT, time.perf_counter() - start)

    print(f"Results written to {args.out}")
    print(checks.report())
    part2, part3 = metrics["part2_state_perturbations"], metrics["part3_constant_residuals"]
    print(f"State perturbations at T={t[-1]:g}: unstable/stable size ratio {part2['final_ratio']:.6g}"
          f" (exact e^2T = {part2['final_ratio_exact']:.6g})")
    print(f"Constant residuals at T={t[-1]:g}: unstable/stable error ratio {part3['final_ratio']:.6g}"
          f" (exact e^T = {part3['final_ratio_exact']:.6g})")
    print("horizon   state ratio (e^2dt)   residual ratio (e^dt)   isotropic")
    for row in metrics["part4_alignment_sweep"]["ratio_table"]:
        print(f"{row['horizon']:7.2f}   {row['state_ratio']:11.4f}           {row['residual_ratio']:11.4f}"
              f"             {row['isotropic_state_ratio']:.4f}")
    return 0 if checks.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
