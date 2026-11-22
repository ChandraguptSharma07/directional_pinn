"""Two-dimensional systems for the dial study (docs/dial_study_plan.md).

Each system has a *dial*: the rate difference lambda_1 - lambda_2 between its
leading (least contracting) direction and its other direction. A residual that
points along the leading direction creates error that persists or grows; a
residual along the other direction is erased within about 1/(rate) time
units. The larger the dial, the more the direction of a residual should matter.

Family L, stiff linear saddle (dial = 1 + kappa):

    dz/dt = A z,   A = R diag(1, -kappa) R^T,

with R the rotation by ``angle``. The expanding eigenvector v_u has rate +1,
the stable eigenvector v_s has rate -kappa. The exact solution is
z(t) = c e^t v_u + d e^(-kappa t) v_s.

Family D, Davis-Skodje slow-fast system (dial = gamma - 1):

    dy1/dt = -y1
    dy2/dt = -gamma y2 + ((gamma - 1) y1 + gamma y1^2) / (1 + y1)^2

The curve y2 = y1 / (1 + y1) is an invariant slow manifold. The Jacobian is
lower triangular with eigenvalues -1 (slow) and -gamma (fast); its
off-diagonal entry makes it non-normal, so singular directions turn as the
state moves.

Each class offers the right-hand side for SciPy (``rhs(t, z)``), batched NumPy
evaluation (``rhs_batch``), the Jacobian (``jacobian``, batched), and a
PyTorch right-hand side (``rhs_torch``) used for the PINN residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Initial-condition windows of docs/dial_study_plan.md, section 3. Window 0 is the pilot.
SADDLE_WINDOWS = {
    0: (1.0, 0.6), 1: (1.0, 0.5), 2: (0.6, -0.5), 3: (1.4, 0.25), 4: (0.8, -1.0),
    5: (1.2, 1.0), 6: (0.7, -0.25), 7: (1.3, -0.75), 8: (0.9, 0.75),
}
DAVIS_SKODJE_WINDOWS = {
    0: (1.25, 0.15), 1: (0.5, 0.1), 2: (1.0, -0.1), 3: (1.5, 0.2), 4: (2.0, -0.2),
    5: (2.5, 0.3), 6: (3.0, -0.3), 7: (3.5, 0.15), 8: (4.0, -0.15),
}


@dataclass(frozen=True)
class StiffSaddle:
    """Linear saddle with expanding rate +1 and contracting rate -kappa (family L)."""

    kappa: float
    angle: float = np.pi / 6

    @property
    def name(self) -> str:
        return f"saddle_k{self.kappa:g}"

    @property
    def dial(self) -> float:
        return 1.0 + self.kappa

    @property
    def unstable_direction(self) -> np.ndarray:
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def stable_direction(self) -> np.ndarray:
        return np.array([-np.sin(self.angle), np.cos(self.angle)])

    @property
    def matrix(self) -> np.ndarray:
        R = np.column_stack([self.unstable_direction, self.stable_direction])
        return R @ np.diag([1.0, -self.kappa]) @ R.T

    def rhs(self, t: float, z: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(z, dtype=float)

    def rhs_batch(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) @ self.matrix.T

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.broadcast_to(self.matrix, z.shape[:-1] + (2, 2)).copy()

    def rhs_torch(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.as_tensor(self.matrix, dtype=x.dtype, device=x.device)
        return x @ A.T

    def initial_state(self, window: int) -> np.ndarray:
        c, d = SADDLE_WINDOWS[window]
        return c * self.unstable_direction + d * self.stable_direction

    def exact_solution(self, t: np.ndarray, window: int) -> np.ndarray:
        c, d = SADDLE_WINDOWS[window]
        t = np.asarray(t, dtype=float)[:, None]
        return c * np.exp(t) * self.unstable_direction + d * np.exp(-self.kappa * t) * self.stable_direction

    def to_dict(self) -> dict:
        return {"family": "L", "kappa": float(self.kappa), "angle": float(self.angle)}


@dataclass(frozen=True)
class DavisSkodje:
    """Davis-Skodje slow-fast system with slow rate -1 and fast rate -gamma (family D)."""

    gamma: float

    @property
    def name(self) -> str:
        return f"davis_skodje_g{self.gamma:g}"

    @property
    def dial(self) -> float:
        return self.gamma - 1.0

    @staticmethod
    def slow_manifold(y1):
        return y1 / (1.0 + y1)

    def _forcing(self, y1):
        return ((self.gamma - 1.0) * y1 + self.gamma * y1**2) / (1.0 + y1) ** 2

    def _forcing_slope(self, y1):
        """d/dy1 of the forcing term: ((gamma - 1) + (gamma + 1) y1) / (1 + y1)^3."""
        return ((self.gamma - 1.0) + (self.gamma + 1.0) * y1) / (1.0 + y1) ** 3

    def rhs_batch(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        y1, y2 = y[..., 0], y[..., 1]
        return np.stack([-y1, -self.gamma * y2 + self._forcing(y1)], axis=-1)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.rhs_batch(y)

    def jacobian(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        J = np.zeros(y.shape[:-1] + (2, 2))
        J[..., 0, 0] = -1.0
        J[..., 1, 0] = self._forcing_slope(y[..., 0])
        J[..., 1, 1] = -self.gamma
        return J

    def rhs_torch(self, x: torch.Tensor) -> torch.Tensor:
        y1, y2 = x[..., 0], x[..., 1]
        return torch.stack([-y1, -self.gamma * y2 + self._forcing(y1)], dim=-1)

    def initial_state(self, window: int) -> np.ndarray:
        y1, offset = DAVIS_SKODJE_WINDOWS[window]
        return np.array([y1, self.slow_manifold(y1) + offset])

    def to_dict(self) -> dict:
        return {"family": "D", "gamma": float(self.gamma)}


DialSystem = StiffSaddle | DavisSkodje

# The dial settings of the plan: rate differences 2, 5, 17 and 65 in both families.
SADDLE_KAPPAS = (1.0, 4.0, 16.0, 64.0)
DAVIS_SKODJE_GAMMAS = (3.0, 6.0, 18.0, 66.0)


def dial_settings() -> list[DialSystem]:
    """All eight settings of the dial study, family L first, each in increasing dial order."""
    return [StiffSaddle(k) for k in SADDLE_KAPPAS] + [DavisSkodje(g) for g in DAVIS_SKODJE_GAMMAS]


def system_from_dict(values: dict) -> DialSystem:
    """Rebuild a system from ``to_dict`` output (used for jobs sent to worker processes)."""
    if values["family"] == "L":
        return StiffSaddle(float(values["kappa"]), float(values.get("angle", np.pi / 6)))
    if values["family"] == "D":
        return DavisSkodje(float(values["gamma"]))
    raise ValueError(f"unknown system family {values['family']!r}")
