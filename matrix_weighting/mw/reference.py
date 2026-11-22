"""Reference solutions and initial conditions (docs/plan.md, sections 5, 7 and 8)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from mw.systems import Lorenz63, LinearSaddle, System

RTOL = 1e-12
ATOL = 1e-12


def reference(system: System, x0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Reference states at ``times`` (must start at 0), shape (len(times), n)."""
    times = np.asarray(times, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if isinstance(system, LinearSaddle):
        c, d = float(system.u @ x0), float(system.s @ x0)  # the eigenvectors are orthonormal
        t = times[:, None]
        return c * np.exp(t) * system.u + d * np.exp(-system.kappa * t) * system.s
    sol = solve_ivp(lambda _t, x: system.rhs(x), (0.0, float(times[-1])), x0, method="DOP853",
                    rtol=RTOL, atol=ATOL, t_eval=times)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def lorenz_attractor_states(count: int = 12, seed: int = 2026112201, spin_up: float = 100.0,
                            span: float = 600.0) -> np.ndarray:
    """``count`` states on the Lorenz-63 attractor, sampled at sorted uniform times in [0, span]."""
    lorenz = Lorenz63()
    start = solve_ivp(lambda _t, x: lorenz.rhs(x), (0.0, spin_up), [1.0, 1.0, 1.0], method="DOP853",
                      rtol=RTOL, atol=ATOL).y[:, -1]
    times = np.sort(np.random.default_rng(seed).uniform(0.0, span, size=count))
    sol = solve_ivp(lambda _t, x: lorenz.rhs(x), (0.0, span), start, method="DOP853", rtol=RTOL, atol=ATOL,
                    t_eval=times)
    return sol.y.T


def attractor_states(system: System, start, count: int = 12, seed: int = 2026112202, spin_up: float = 100.0,
                     span: float = 600.0) -> np.ndarray:
    """``count`` attractor states of any system: spin up from ``start``, then sorted uniform times in [0, span]."""
    s0 = solve_ivp(lambda _t, x: system.rhs(x), (0.0, spin_up), np.asarray(start, float), method="DOP853",
                   rtol=RTOL, atol=ATOL).y[:, -1]
    times = np.sort(np.random.default_rng(seed).uniform(0.0, span, size=count))
    return solve_ivp(lambda _t, x: system.rhs(x), (0.0, span), s0, method="DOP853", rtol=RTOL, atol=ATOL,
                     t_eval=times).y.T


def long_run_range(system: System, start, spin_up: float = 100.0, span: float = 600.0) -> tuple[np.ndarray, np.ndarray]:
    """Componentwise min and max along a long reference trajectory (for output scaling)."""
    s0 = solve_ivp(lambda _t, x: system.rhs(x), (0.0, spin_up), np.asarray(start, float), method="DOP853",
                   rtol=RTOL, atol=ATOL).y[:, -1]
    y = solve_ivp(lambda _t, x: system.rhs(x), (0.0, span), s0, method="DOP853", rtol=1e-10, atol=1e-10,
                  t_eval=np.linspace(0, span, 60001)).y.T
    return y.min(axis=0), y.max(axis=0)


def damped_initial_state(lorenz_state: np.ndarray, k: int) -> np.ndarray:
    """(x, y, z, q_1..q_k) with q_j(0) = x(0) (docs/plan.md, section 9)."""
    x = np.asarray(lorenz_state, dtype=float)
    return np.concatenate([x, np.full(k, x[0])])
