"""Tangent dynamics: how small perturbations evolve along a trajectory (Phase 3).

A small perturbation delta of a trajectory x(t) evolves, to first order, by the
tangent equation

    d delta / dt = J(x(t)) delta,

so delta(t) = Phi(t, s) delta(s), where the propagator Phi(t, s) solves

    d Phi(t, s) / dt = J(x(t)) Phi(t, s),   Phi(s, s) = I.

``step_propagators`` computes Phi(t_{i+1}, t_i) for every step of a uniform time
grid. Propagators over longer intervals are products of step propagators,

    Phi(t_{j+m}, t_j) = Phi(t_{j+m}, t_{j+m-1}) ... Phi(t_{j+1}, t_j),

which ``propagators_over`` evaluates for every start j at once. Multiplying
short, well-conditioned matrices avoids inverting long-time propagators, whose
condition number grows like exp((lambda_1 - lambda_3) t), about exp(15.5 t)
for Lorenz-63.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


def step_propagators(
    jacobian_along: Callable[[np.ndarray], np.ndarray],
    t_grid: np.ndarray,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> np.ndarray:
    """Phi(t_{i+1}, t_i) for every step of a uniform grid, shape (n_steps, d, d).

    Args:
        jacobian_along: maps an array of times to the Jacobians J(x(t)) along
            the trajectory, shape (..., d, d).
        t_grid: uniform time grid.

    All steps are integrated together in local time s in [0, h]: step i solves
    dM_i/ds = J(x(t_i + s)) M_i with M_i(0) = I, using DOP853.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    h = t_grid[1] - t_grid[0]
    if not np.allclose(np.diff(t_grid), h, rtol=1e-9, atol=0.0):
        raise ValueError("t_grid must be uniform")
    starts = t_grid[:-1]
    d = jacobian_along(starts[:1]).shape[-1]
    n = starts.size

    def rhs(s, flat):
        M = flat.reshape(n, d, d)
        return (jacobian_along(starts + s) @ M).ravel()

    initial = np.broadcast_to(np.eye(d), (n, d, d)).ravel()
    sol = solve_ivp(rhs, (0.0, h), initial, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"tangent integration failed: {sol.message}")
    return sol.y[:, -1].reshape(n, d, d)


def propagators_over(step_props: np.ndarray, m: int) -> np.ndarray:
    """Phi(t_{j+m}, t_j) for j = 0, ..., n - m, shape (n - m + 1, d, d), from n step propagators.

    Uses repeated doubling, P_{k+1}[j] = P_k[j + 2^k] @ P_k[j], so the cost is
    O(n log m) batched 3x3 products instead of O(n m).
    """
    n = len(step_props)
    if not 1 <= m <= n:
        raise ValueError(f"need 1 <= m <= {n}, got {m}")
    power, length = step_props, 1  # power[j] = Phi(t_{j+length}, t_j)
    result, done = None, 0  # result[j] = Phi(t_{j+done}, t_j)
    remaining = m
    while True:
        if remaining & 1:
            if result is None:
                result, done = power, length
            else:
                count = n - done - length + 1
                result = power[done : done + count] @ result[:count]
                done += length
        remaining >>= 1
        if not remaining:
            return result[: n - m + 1]
        count = n - 2 * length + 1
        power = power[length : length + count] @ power[:count]
        length *= 2


def integrated_propagators(step_props: np.ndarray, m: int, h: float) -> np.ndarray:
    """K(t_j; m h) = integral of Phi(t_{j+m}, s) ds over s in [t_j, t_{j+m}], for j = 0, ..., n - m.

    K maps a constant residual r on [t_j, t_{j+m}] to the (linearized) error it
    creates by t_{j+m}: e_created = K r. Computed by the trapezoidal rule on the
    grid, with Phi(t_k, t_{k-i}) built for every end index k at once by the
    recursion Phi(t_k, t_{k-i}) = Phi(t_k, t_{k-i+1}) Phi(t_{k-i+1}, t_{k-i}).
    Returns shape (n - m + 1, d, d), indexed by the start j.
    """
    n, d, _ = step_props.shape
    if not 1 <= m <= n:
        raise ValueError(f"need 1 <= m <= {n}, got {m}")
    ends = np.arange(m, n + 1)  # end grid index k of every interval
    P = np.broadcast_to(np.eye(d), (len(ends), d, d)).copy()  # Phi(t_k, t_k)
    total = 0.5 * P
    for i in range(1, m + 1):
        P = P @ step_props[ends - i]  # Phi(t_k, t_{k-i})
        total += (0.5 if i == m else 1.0) * P
    return h * total


def pinn_path_step_propagators(t: np.ndarray, states: np.ndarray,
                               jacobian: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Step propagators along an approximate trajectory, such as a PINN's own output.

    ``states`` are the approximate states on the uniform grid ``t``; between grid
    points they are taken from a cubic spline. No reference solution is used, so
    the result is a reference-free estimate of Phi(t_{i+1}, t_i).
    """
    from scipy.interpolate import CubicSpline

    spline = CubicSpline(np.asarray(t, dtype=float), np.asarray(states, dtype=float), axis=0)
    return step_propagators(lambda times: jacobian(spline(times)), t)


def frozen_jacobian_propagators(states: np.ndarray, jacobian: Callable[[np.ndarray], np.ndarray],
                                dt: float) -> np.ndarray:
    """expm(J(x) dt) for every state x (shape (N, d) -> (N, d, d)): the Jacobian held fixed over dt."""
    from scipy.linalg import expm

    return expm(jacobian(np.asarray(states, dtype=float)) * dt)
