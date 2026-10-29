"""Fully connected PINN for an initial value problem: time in, state out (Phase 4).

The network sees a rescaled time and produces a rescaled state:

    tau = 2 t / T - 1                      (input in [-1, 1] on the window [0, T])
    x_hat(t) = mean + scale * net(tau)     (outputs of order one inside the network)

Both scalings are fixed affine maps, so x_hat is still an ordinary
differentiable function of t and its time derivative (needed for the residual)
follows by the chain rule inside automatic differentiation.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


ACTIVATIONS = {"tanh": nn.Tanh, "sin": Sine, "gelu": nn.GELU}


class PINN(nn.Module):
    """MLP t -> x_hat(t) with fixed input and output scaling."""

    def __init__(
        self,
        t_final: float,
        out_mean: Sequence[float],
        out_scale: Sequence[float],
        hidden: Sequence[int] = (64, 64, 64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        layers: list[nn.Module] = []
        width_in = 1
        for width in hidden:
            linear = nn.Linear(width_in, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, ACTIVATIONS[activation]()]
            width_in = width
        last = nn.Linear(width_in, len(out_mean))
        nn.init.xavier_normal_(last.weight)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)
        self.t_final = float(t_final)
        self.register_buffer("out_mean", torch.tensor(out_mean, dtype=torch.get_default_dtype()))
        self.register_buffer("out_scale", torch.tensor(out_scale, dtype=torch.get_default_dtype()))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t has shape (N, 1); returns x_hat with shape (N, d)."""
        tau = 2.0 * t / self.t_final - 1.0
        return self.out_mean + self.out_scale * self.net(tau)
