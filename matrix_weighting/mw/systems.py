"""Dynamical systems for the matrix-weighting study (docs/plan.md).

Every system gives, for a batch of states x with shape (..., n):

    rhs(x)  -> f(x), shape (..., n)
    jac(x)  -> J(x) = df/dx, shape (..., n, n)

Both work on NumPy arrays (float64, references and weights) and on PyTorch
tensors (the PINN residual, and weights computed along the PINN's own path).

The two dial families copy the definitions of src/systems/dial_systems.py so
that this study does not depend on code another study may still change.
"""

from __future__ import annotations

import numpy as np
import torch


def _stack(arrays, like):
    if isinstance(like, torch.Tensor):
        return torch.stack(arrays, dim=-1)
    return np.stack(arrays, axis=-1)


def _zeros(like, shape):
    if isinstance(like, torch.Tensor):
        return like.new_zeros(shape)
    return np.zeros(shape, dtype=float)


def _const(value, like):
    """A constant matrix/vector in the array type and precision of ``like``."""
    if isinstance(like, torch.Tensor):
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    return np.asarray(value, dtype=float)


class System:
    """Base class: an autonomous ODE dx/dt = f(x) in n dimensions."""

    dim: int
    name: str

    def rhs(self, x):
        raise NotImplementedError

    def jac(self, x):
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------- dial family L
SADDLE_WINDOWS = {
    0: (1.0, 0.6), 1: (1.0, 0.5), 2: (0.6, -0.5), 3: (1.4, 0.25), 4: (0.8, -1.0),
    5: (1.2, 1.0), 6: (0.7, -0.25), 7: (1.3, -0.75), 8: (0.9, 0.75),
}


class LinearSaddle(System):
    """dz/dt = A z with A = R diag(1, -kappa) R^T (the dial study's family L)."""

    dim = 2

    def __init__(self, kappa: float, angle: float = np.pi / 6) -> None:
        self.kappa = float(kappa)
        self.angle = float(angle)
        self.u = np.array([np.cos(angle), np.sin(angle)])  # expanding eigenvector, rate +1
        self.s = np.array([-np.sin(angle), np.cos(angle)])  # contracting eigenvector, rate -kappa
        R = np.column_stack([self.u, self.s])
        self.A = R @ np.diag([1.0, -self.kappa]) @ R.T
        self.name = f"saddle_k{self.kappa:g}"

    def rhs(self, x):
        return x @ _const(self.A.T, x)

    def jac(self, x):
        A = _const(self.A, x)
        if isinstance(x, torch.Tensor):
            return A.expand(*x.shape[:-1], 2, 2).clone()
        return np.broadcast_to(A, x.shape[:-1] + (2, 2)).copy()

    def initial_state(self, window: int) -> np.ndarray:
        c, d = SADDLE_WINDOWS[window]
        return c * self.u + d * self.s

    def exact(self, t: np.ndarray, window: int) -> np.ndarray:
        c, d = SADDLE_WINDOWS[window]
        t = np.asarray(t, dtype=float)[:, None]
        return c * np.exp(t) * self.u + d * np.exp(-self.kappa * t) * self.s

    def to_dict(self) -> dict:
        return {"kind": "saddle", "kappa": self.kappa, "angle": self.angle}


# ---------------------------------------------------------------- dial family D
DAVIS_SKODJE_WINDOWS = {
    0: (1.25, 0.15), 1: (0.5, 0.1), 2: (1.0, -0.1), 3: (1.5, 0.2), 4: (2.0, -0.2),
    5: (2.5, 0.3), 6: (3.0, -0.3), 7: (3.5, 0.15), 8: (4.0, -0.15),
}


class DavisSkodje(System):
    """dy1/dt = -y1, dy2/dt = -gamma y2 + ((gamma-1) y1 + gamma y1^2)/(1+y1)^2 (family D)."""

    dim = 2

    def __init__(self, gamma: float) -> None:
        self.gamma = float(gamma)
        self.name = f"davis_skodje_g{self.gamma:g}"

    def _forcing(self, y1):
        return ((self.gamma - 1.0) * y1 + self.gamma * y1**2) / (1.0 + y1) ** 2

    def _forcing_slope(self, y1):
        return ((self.gamma - 1.0) + (self.gamma + 1.0) * y1) / (1.0 + y1) ** 3

    def rhs(self, x):
        y1, y2 = x[..., 0], x[..., 1]
        return _stack([-y1, -self.gamma * y2 + self._forcing(y1)], x)

    def jac(self, x):
        J = _zeros(x, x.shape[:-1] + (2, 2))
        J[..., 0, 0] = -1.0
        J[..., 1, 0] = self._forcing_slope(x[..., 0])
        J[..., 1, 1] = -self.gamma
        return J

    def initial_state(self, window: int) -> np.ndarray:
        y1, offset = DAVIS_SKODJE_WINDOWS[window]
        return np.array([y1, y1 / (1.0 + y1) + offset])

    def to_dict(self) -> dict:
        return {"kind": "davis_skodje", "gamma": self.gamma}


# ---------------------------------------------------------------- Lorenz family
class Lorenz63(System):
    """Lorenz-63, optionally driving k damped modes dq_j/dt = gamma_j (x - q_j).

    The modes never act back on (x, y, z), so the Lyapunov spectrum is the
    Lorenz spectrum plus {-gamma_j}. k = 0 is plain Lorenz-63.
    """

    def __init__(self, k: int = 0, gamma_min: float = 5.0, gamma_max: float = 50.0,
                 sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> None:
        self.k = int(k)
        self.sigma, self.rho, self.beta = float(sigma), float(rho), float(beta)
        self.gamma_min, self.gamma_max = float(gamma_min), float(gamma_max)
        self.gammas = np.geomspace(gamma_min, gamma_max, self.k) if self.k > 0 else np.zeros(0)
        self.dim = 3 + self.k
        self.name = "lorenz63" if self.k == 0 else f"lorenz63_damped{self.k}"

    def rhs(self, x):
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        core = [self.sigma * (Y - X), X * (self.rho - Z) - Y, X * Y - self.beta * Z]
        if self.k == 0:
            return _stack(core, x)
        g = _const(self.gammas, x)
        modes = g * (X[..., None] - x[..., 3:])
        if isinstance(x, torch.Tensor):
            return torch.cat([torch.stack(core, dim=-1), modes], dim=-1)
        return np.concatenate([np.stack(core, axis=-1), modes], axis=-1)

    def jac(self, x):
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        J = _zeros(x, x.shape[:-1] + (self.dim, self.dim))
        J[..., 0, 0] = -self.sigma
        J[..., 0, 1] = self.sigma
        J[..., 1, 0] = self.rho - Z
        J[..., 1, 1] = -1.0
        J[..., 1, 2] = -X
        J[..., 2, 0] = Y
        J[..., 2, 1] = X
        J[..., 2, 2] = -self.beta
        for j, g in enumerate(self.gammas):
            J[..., 3 + j, 0] = g
            J[..., 3 + j, 3 + j] = -g
        return J

    def to_dict(self) -> dict:
        return {"kind": "lorenz", "k": self.k, "gamma_min": self.gamma_min, "gamma_max": self.gamma_max,
                "sigma": self.sigma, "rho": self.rho, "beta": self.beta}


def system_from_dict(d: dict) -> System:
    kind = d["kind"]
    if kind == "saddle":
        return LinearSaddle(d["kappa"], d.get("angle", np.pi / 6))
    if kind == "davis_skodje":
        return DavisSkodje(d["gamma"])
    if kind == "lorenz":
        return Lorenz63(d.get("k", 0), d.get("gamma_min", 5.0), d.get("gamma_max", 50.0),
                        d.get("sigma", 10.0), d.get("rho", 28.0), d.get("beta", 8.0 / 3.0))
    raise ValueError(f"unknown system kind {kind!r}")
