"""Lorenz-63 system: right-hand side, analytic Jacobian and fixed points (Phase 1).

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

The classical chaotic parameters are sigma = 10, rho = 28, beta = 8/3
(Lorenz, J. Atmos. Sci. 20, 1963).

Array conventions: a state has shape (3,), a batch of states has shape (N, 3)
with one state per row, and the Jacobian of a batch has shape (N, 3, 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.dynamics.jacobian import JacobianCheck, check_jacobian


@dataclass(frozen=True)
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    @classmethod
    def from_dict(cls, values: dict) -> LorenzParams:
        """Build from a config mapping such as {"sigma": 10, "rho": 28, "beta": 8/3}; unknown keys raise TypeError."""
        return cls(**{key: float(value) for key, value in values.items()})


def lorenz_rhs(t: float, state: np.ndarray, params: LorenzParams) -> np.ndarray:
    """Right-hand side f(state) of the Lorenz equations.

    The system is autonomous, so t is ignored; it is accepted so that the
    function has the (t, y) signature of scipy's solve_ivp. Works for one state
    (3,) or a batch (N, 3), and for complex input (used by the complex-step check).
    """
    state = np.asarray(state)
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    return np.stack(
        [
            params.sigma * (y - x),  # dx/dt
            x * (params.rho - z) - y,  # dy/dt
            x * y - params.beta * z,  # dz/dt
        ],
        axis=-1,
    )


def lorenz_jacobian(state: np.ndarray, params: LorenzParams) -> np.ndarray:
    """Analytic Jacobian J = df/d(state):

        J(x, y, z) = [[ -sigma,   sigma,   0    ],
                      [ rho - z,  -1,      -x   ],
                      [ y,        x,       -beta]]

    Row i holds the partial derivatives of equation i with respect to (x, y, z).
    Shape (3, 3) for one state, (N, 3, 3) for a batch.
    """
    state = np.asarray(state, dtype=float)
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    J = np.empty(state.shape[:-1] + (3, 3))
    J[..., 0, 0] = -params.sigma
    J[..., 0, 1] = params.sigma
    J[..., 0, 2] = 0.0
    J[..., 1, 0] = params.rho - z
    J[..., 1, 1] = -1.0
    J[..., 1, 2] = -x
    J[..., 2, 0] = y
    J[..., 2, 1] = x
    J[..., 2, 2] = -params.beta
    return J


def lorenz_fixed_points(params: LorenzParams) -> dict[str, np.ndarray]:
    """Equilibria, where all three right-hand sides vanish.

    dx/dt = 0 gives y = x, and dz/dt = 0 gives x^2 = beta z. Then
    dy/dt = x (rho - 1 - z) = 0, so either x = 0 (the origin) or z = rho - 1.
    For rho > 1 the second case gives C+ and C- = (+-a, +-a, rho - 1) with
    a = sqrt(beta (rho - 1)).
    """
    points = {"origin": np.zeros(3)}
    if params.rho > 1.0:
        a = np.sqrt(params.beta * (params.rho - 1.0))
        points["C+"] = np.array([a, a, params.rho - 1.0])
        points["C-"] = np.array([-a, -a, params.rho - 1.0])
    return points


def lorenz_divergence(params: LorenzParams) -> float:
    """Trace of the Jacobian, -(sigma + 1 + beta), which is the same at every state.

    It is the rate at which the flow changes small volumes, d/dt log(volume) = trace J.
    Over long times it also equals the sum of the three Lyapunov exponents.
    """
    return -(params.sigma + 1.0 + params.beta)


# Sampling box for the Jacobian check: it contains the attractor with a wide margin.
JACOBIAN_CHECK_BOX = ((-30.0, 30.0), (-30.0, 30.0), (-5.0, 60.0))
JACOBIAN_TOLERANCES = {"complex_step": 1e-12, "central_difference": 1e-7}


def verify_lorenz_jacobian(
    params: LorenzParams,
    jacobian: Callable[[np.ndarray, LorenzParams], np.ndarray] = lorenz_jacobian,
    n_states: int = 1000,
    seed: int = 0,
    box=JACOBIAN_CHECK_BOX,
) -> JacobianCheck:
    """Compare a Lorenz Jacobian with complex-step and central-difference Jacobians.

    The states are the fixed points plus n_states uniform random states in
    `box`. The box extends beyond the attractor on purpose: PINN predictions
    need not lie on the attractor, and the formula must hold everywhere.

    Tolerances on the relative error:
      complex step         1e-12  the method is exact up to rounding (~1e-16)
      central differences  1e-7   each Lorenz equation is affine in each variable
                                  separately (the only products are x z and x y),
                                  so the differences have no truncation error;
                                  only rounding, of order 1e-10, remains

    Later phases call this before using the Jacobian and stop if it fails.
    """
    rng = np.random.default_rng(seed)
    low, high = np.asarray(box, dtype=float).T
    states = np.vstack([*lorenz_fixed_points(params).values(), rng.uniform(low, high, size=(n_states, 3))])
    return check_jacobian(
        lambda state: lorenz_rhs(0.0, state, params),
        lambda state: jacobian(state, params),
        states,
        JACOBIAN_TOLERANCES,
    )
