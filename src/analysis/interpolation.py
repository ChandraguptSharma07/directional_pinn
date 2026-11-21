"""Coarse residual sampling, interpolation, and trailing averages.

Linear interpolation is applied to the vector residual before taking its norm;
the sample-and-hold reconstruction provides a coarse-sampling control. At times
between sampled knots, linear interpolation uses the next knot, so it is an
offline diagnostic, not a real-time causal estimator.
"""

from __future__ import annotations

import numpy as np


def reconstruct_residual(t: np.ndarray, residual: np.ndarray, stride: int, method: str) -> np.ndarray:
    """Reconstruct a vector residual on ``t`` from every ``stride``-th sample.

    The final endpoint is included even when it is not a multiple of stride.
    ``method`` is either ``linear`` or ``hold`` (last observed sample).
    """
    t = np.asarray(t, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if t.ndim != 1 or residual.ndim != 2 or len(t) != len(residual) or len(t) < 2:
        raise ValueError("expected matching time (n,) and residual (n, d) arrays")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(residual)) or np.any(np.diff(t) <= 0):
        raise ValueError("times must increase and all inputs must be finite")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if method not in {"linear", "hold"}:
        raise ValueError("method must be 'linear' or 'hold'")

    indices = np.arange(0, len(t), stride)
    if indices[-1] != len(t) - 1:
        indices = np.r_[indices, len(t) - 1]
    coarse_t, coarse_r = t[indices], residual[indices]
    if method == "hold":
        source = np.searchsorted(coarse_t, t, side="right") - 1
        return coarse_r[source]
    return np.column_stack([np.interp(t, coarse_t, coarse_r[:, d]) for d in range(residual.shape[1])])


def trailing_mean_magnitude(t: np.ndarray, residual: np.ndarray, width: float) -> np.ndarray:
    """Mean of ``|residual|`` over [t-width, t] on a uniform grid.

    Returns NaN before a complete trailing window exists. The trapezoidal
    integral of reconstructed residual magnitudes is exact for piecewise linear
    magnitude values on each fine-grid cell.
    """
    t = np.asarray(t, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if t.ndim != 1 or residual.ndim != 2 or len(t) != len(residual) or len(t) < 2:
        raise ValueError("expected matching time (n,) and residual (n, d) arrays")
    h = t[1] - t[0]
    if h <= 0 or not np.allclose(np.diff(t), h, rtol=1e-9, atol=1e-12):
        raise ValueError("time grid must be uniform and increasing")
    m = int(round(width / h))
    if m < 1 or not np.isclose(m * h, width, rtol=1e-9, atol=1e-12):
        raise ValueError("width must be a positive multiple of grid spacing")
    magnitude = np.linalg.norm(residual, axis=1)
    cumulative = np.r_[0.0, np.cumsum(0.5 * h * (magnitude[:-1] + magnitude[1:]))]
    out = np.full(len(t), np.nan)
    out[m:] = (cumulative[m:] - cumulative[:-m]) / width
    return out
