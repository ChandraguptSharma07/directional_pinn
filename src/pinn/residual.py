"""Differential-equation residual of a PINN (Phase 4).

For a network x_hat(t) and the ODE dx/dt = f(x), the residual is

    r(t) = d x_hat / dt - f(x_hat(t)),

computed exactly (up to rounding) by automatic differentiation. For Lorenz-63:

    r_x = dx_hat/dt - sigma (y_hat - x_hat)
    r_y = dy_hat/dt - (x_hat (rho - z_hat) - y_hat)
    r_z = dz_hat/dt - (x_hat y_hat - beta z_hat)
"""

from __future__ import annotations

from typing import Callable

import torch

from src.systems.lorenz63 import LorenzParams


def lorenz_rhs_torch(x: torch.Tensor, params: LorenzParams) -> torch.Tensor:
    """Lorenz right-hand side for a batch of states, shape (N, 3) -> (N, 3)."""
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    return torch.stack(
        [
            params.sigma * (Y - X),
            X * (params.rho - Z) - Y,
            X * Y - params.beta * Z,
        ],
        dim=1,
    )


def state_and_time_derivative(
    model: Callable[[torch.Tensor], torch.Tensor], t: torch.Tensor, create_graph: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """x_hat(t) and d x_hat/dt at times t (shape (N, 1)); both have shape (N, d).

    Each output component is differentiated with respect to t separately. Every
    x_hat(t_n) depends only on its own t_n, so the gradient of the sum over n
    gives all N derivatives at once. With create_graph=True the result can be
    differentiated again (needed for training).
    """
    t = t.detach().requires_grad_(True)
    x = model(t)
    columns = [
        torch.autograd.grad(x[:, i].sum(), t, create_graph=create_graph, retain_graph=True)[0]
        for i in range(x.shape[1])
    ]
    return x, torch.cat(columns, dim=1)


def ode_residual(
    model: Callable[[torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    rhs: Callable[[torch.Tensor], torch.Tensor],
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (x_hat, d x_hat/dt, residual) at times t for the ODE dx/dt = rhs(x)."""
    x, dxdt = state_and_time_derivative(model, t, create_graph=create_graph)
    return x, dxdt, dxdt - rhs(x)


def lorenz_residual(
    model: Callable[[torch.Tensor], torch.Tensor],
    t: torch.Tensor,
    params: LorenzParams,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``ode_residual`` for the Lorenz equations."""
    return ode_residual(model, t, lambda x: lorenz_rhs_torch(x, params), create_graph=create_graph)
