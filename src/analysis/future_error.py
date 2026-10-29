"""Future error of a PINN and its decomposition (Phase 5).

On a uniform grid t_j = j h with error vectors e_j = x_hat(t_j) - x(t_j):

    E_Delta(t_j) = |e_{j+m}|,  m = Delta / h
        the future error after a horizon Delta;
    N_Delta(t_j) = |e_{j+m} - Phi(t_{j+m}, t_j) e_j|
        the part of it created inside [t_j, t_j + Delta]: the future error minus
        the linear propagation of the error already present at t_j.

The linear (Duhamel) reconstruction of the whole error from its initial value
and the residual,

    e(t) ~ Phi(t, 0) e(0) + integral_0^t Phi(t, s) r(s) ds,

is evaluated with the one-step recursion
    e_{k+1} = Phi_k e_k + (h / 2) (Phi_k r_k + r_{k+1})
(trapezoidal rule for the integral over one step).
"""

from __future__ import annotations

import numpy as np


def horizon_steps(horizon: float, h: float) -> int:
    """Number of grid steps m with m h = horizon; raises if the horizon is not a multiple of h."""
    m = int(round(horizon / h))
    if m < 1 or not np.isclose(m * h, horizon, rtol=1e-9, atol=1e-12):
        raise ValueError(f"horizon {horizon} is not a positive multiple of the step {h}")
    return m


def future_error(error: np.ndarray, m: int) -> np.ndarray:
    """|e_{j+m}| for j = 0, ..., N - 1 - m (aligned with the first N - m grid times)."""
    return np.linalg.norm(error[m:], axis=1)


def created_error(error: np.ndarray, propagators: np.ndarray, m: int) -> np.ndarray:
    """|e_{j+m} - Phi(t_{j+m}, t_j) e_j| for j = 0, ..., N - 1 - m.

    `propagators` must hold Phi(t_{j+m}, t_j) for at least these j.
    """
    n = len(error) - m
    propagated = np.einsum("jab,jb->ja", propagators[:n], error[:n])
    return np.linalg.norm(error[m:] - propagated, axis=1)


def linearized_error(e0: np.ndarray, residual: np.ndarray, step_props: np.ndarray, h: float) -> np.ndarray:
    """Duhamel reconstruction of the error on the grid, shape (N, d), from e(0) and the residual."""
    n = len(residual)
    out = np.empty_like(residual)
    out[0] = e0
    for k in range(n - 1):
        out[k + 1] = step_props[k] @ out[k] + 0.5 * h * (step_props[k] @ residual[k] + residual[k + 1])
    return out
