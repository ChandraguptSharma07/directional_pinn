"""Splitting a residual vector into a part along a chosen direction and the rest.

For residual vectors r (one per time point) and unit vectors u (for example the
local expanding direction at each time):

    r_u    = (r . u) u              part of r along u
    r_perp = r - r_u                part of r perpendicular to u
    R      = |r|                    ordinary residual size          (Metric A)
    R_u    = |r . u|                residual along u                (Metric B)
    R_perp = |r_perp|               transverse residual             (Metric C)
    A      = |r . u| / (|r| + eps)  alignment, between 0 and 1

By construction r = r_u + r_perp and R^2 = R_u^2 + R_perp^2 (Pythagoras), and
only |r . u| appears, so u and -u give identical results. The sign must not
matter: Lyapunov vectors are only defined up to sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class ResidualDecomposition:
    """Decomposition of residuals relative to directions.

    Vector fields (r_u, r_perp) have the broadcast shape of the inputs, (N, d) or
    (d,); scalar fields (R, R_u, R_perp, alignment) drop the last axis.
    """

    r_u: np.ndarray
    r_perp: np.ndarray
    R: np.ndarray
    R_u: np.ndarray
    R_perp: np.ndarray
    alignment: np.ndarray


def decompose_residual(r, u, eps: float = 1e-12, unit_tol: float = 1e-6) -> ResidualDecomposition:
    """Decompose residuals r relative to unit directions u.

    Args:
        r: residual vectors, shape (N, d) or (d,).
        u: unit directions, shape (N, d) (one per residual) or (d,) (one for all).
        eps: added to |r| in the alignment only, to avoid 0/0 when r = 0. Keep it
            far below every residual size of interest.
        unit_tol: allowed deviation of |u| from 1.

    Raises:
        ValueError: if some u does not have unit length.
    """
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)
    if np.any(np.abs(np.linalg.norm(u, axis=-1) - 1.0) > unit_tol):
        raise ValueError("directions u must have unit length")
    dot = np.asarray(np.sum(r * u, axis=-1))  # r . u
    r_u = dot[..., None] * u
    r_perp = r - r_u
    R = np.linalg.norm(r, axis=-1)
    R_u = np.abs(dot)
    return ResidualDecomposition(
        r_u=r_u,
        r_perp=r_perp,
        R=R,
        R_u=R_u,
        R_perp=np.linalg.norm(r_perp, axis=-1),
        alignment=R_u / (R + eps),
    )
