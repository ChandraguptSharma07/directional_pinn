"""Lyapunov exponents and Lyapunov-type directions from step propagators (Phase 3).

Two kinds of "expanding direction" are computed, and they answer different
questions (see docs/phase3_lyapunov.md and Phase 0, section 4):

backward (Gram-Schmidt) Lyapunov vectors b_i(t)
    Columns of Q(t) in the QR method: propagate an orthonormal basis forward
    and re-orthonormalise it with QR after every step,
        Phi(t_{i+1}, t_i) Q_i = Q_{i+1} R_i   (R_i upper triangular, diag > 0).
    After a spin-up, b_1(t) is the direction into which perturbations from the
    past have turned. It depends only on the past of the trajectory. The time
    averages of log R_i[k, k] are the Lyapunov exponents (Benettin et al. 1980;
    Shimada & Nagashima 1979).
finite-time singular vectors s_i(t; dt)
    Right singular vectors of Phi(t + dt, t): s_1 is the unit perturbation at
    time t that grows the most over the next dt, by the factor sigma_1. They
    depend only on the trajectory between t and t + dt.

Neither is a covariant Lyapunov vector, except that b_1 coincides with the
first covariant vector once converged.
"""

from __future__ import annotations

import numpy as np


def qr_iteration(step_props: np.ndarray, Q0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR method along a trajectory.

    Returns:
        Q: orthonormal bases at every grid time, shape (n_steps + 1, d, d);
            Q[i][:, k] is the k-th backward vector at t_i (column k).
        log_stretch: log of the diagonal of R_i, shape (n_steps, d): the
            growth of the k-th direction during step i, after removing the
            parts along directions 1, ..., k-1.
    """
    n, d, _ = step_props.shape
    Q = np.empty((n + 1, d, d))
    log_stretch = np.empty((n, d))
    Q[0] = Q0
    for i in range(n):
        q, r = np.linalg.qr(step_props[i] @ Q[i])
        signs = np.sign(np.diag(r))  # make the diagonal of R positive, so Q changes continuously
        Q[i + 1] = q * signs
        log_stretch[i] = np.log(np.abs(np.diag(r)))
    return Q, log_stretch


def lyapunov_exponents(log_stretch: np.ndarray, h: float) -> np.ndarray:
    """Time-averaged exponents: sum of log stretch factors divided by the elapsed time."""
    return log_stretch.sum(axis=0) / (h * len(log_stretch))


def running_exponents(log_stretch: np.ndarray, h: float) -> np.ndarray:
    """Exponent estimates after each step, shape (n_steps, d)."""
    steps = np.arange(1, len(log_stretch) + 1)[:, None]
    return np.cumsum(log_stretch, axis=0) / (h * steps)


def singular_directions(propagators: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Singular values and right singular vectors of a batch of propagators.

    Returns:
        sigma: shape (N, d), in decreasing order; sigma[:, k] is the growth
            factor of the k-th direction over the propagator's interval.
        V: shape (N, d, d); V[j][:, k] is the k-th right singular vector s_k.
    """
    _, sigma, vt = np.linalg.svd(propagators)
    return sigma, np.swapaxes(vt, -1, -2)


def make_continuous(vectors: np.ndarray) -> np.ndarray:
    """Flip signs of unit vectors (shape (N, d)) so consecutive ones point the same way.

    Lyapunov and singular vectors are only defined up to sign; this matters for
    plots, not for |r . u|.
    """
    out = vectors.copy()
    for i in range(1, len(out)):
        if out[i] @ out[i - 1] < 0:
            out[i] = -out[i]
    return out


def line_misalignment(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sine of the angle between the lines spanned by unit vectors u and v (shape (..., d)).

    Computed as |u - (u . v) v|, which resolves small angles down to rounding
    level; sqrt(1 - cos^2) cannot resolve angles below about 2e-8.
    """
    cos = np.sum(u * v, axis=-1, keepdims=True)
    return np.linalg.norm(u - cos * v, axis=-1)
