"""A batch of independent PINNs trained together (docs/plan.md, section 5).

Network b maps time to state:

    tau = 2 t / T - 1
    raw_b(t) = mean + scale * net_b(tau)          (4 x 64 tanh MLP by default)
    x_hat_b(t) = x0_b + (t / T) (raw_b(t) - x0_b)  (exact initial state)

The batch shares the time grid but no parameters: every weight tensor has a
leading batch axis, and the loss is a sum of per-network losses, so with Adam
(which acts on each parameter entry separately) the networks train exactly as
if they were trained one at a time.

The time derivative is propagated forward through the layers together with
the values (forward-mode differentiation by hand):

    z = W h + b,  dz = W dh,  h' = tanh(z),  dh' = (1 - h'^2) dz.

This gives dx_hat/dt exactly, without a second autograd pass.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


def init_seed(seed: int, ic_id: int) -> int:
    """Initialization seed of the network for one (seed, initial condition) pair."""
    return 1000 * int(seed) + int(ic_id)


class BatchedPINN(nn.Module):
    def __init__(self, x0: np.ndarray, t_final: float, out_mean: Sequence[float], out_scale: Sequence[float],
                 init_seeds: Sequence[int], hidden: Sequence[int] = (64, 64, 64, 64),
                 dtype: torch.dtype = torch.float32, hard_ic: bool = True) -> None:
        super().__init__()
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim != 2 or len(init_seeds) != x0.shape[0]:
            raise ValueError("x0 must have shape (batch, n) with one init seed per network")
        self.batch, self.dim = x0.shape
        self.t_final = float(t_final)
        self.hard_ic = bool(hard_ic)  # False: x_hat = raw, and the loss must penalize x_hat(0) - x0
        widths = [1, *hidden, self.dim]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            self.weights.append(nn.Parameter(torch.empty(self.batch, fan_out, fan_in, dtype=dtype)))
            self.biases.append(nn.Parameter(torch.zeros(self.batch, fan_out, dtype=dtype)))
        # Xavier-normal initialization, drawn per network from its own generator so that a
        # network's start does not depend on which other networks share the batch.
        with torch.no_grad():
            for b, s in enumerate(init_seeds):
                g = torch.Generator().manual_seed(int(s))
                for W in self.weights:
                    fan_out, fan_in = W.shape[1], W.shape[2]
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    W[b].copy_(torch.randn(fan_out, fan_in, generator=g, dtype=torch.float64).to(dtype) * std)
        self.register_buffer("x0", torch.tensor(x0, dtype=dtype))
        self.register_buffer("out_mean", torch.tensor(np.asarray(out_mean, dtype=float), dtype=dtype))
        self.register_buffer("out_scale", torch.tensor(np.asarray(out_scale, dtype=float), dtype=dtype))

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """t: shared times, shape (N,). Returns x_hat and dx_hat/dt, each (batch, N, n)."""
        T = self.t_final
        tau = (2.0 * t / T - 1.0).reshape(1, -1, 1).expand(self.batch, -1, 1)
        h = tau
        dh = torch.full_like(tau, 2.0 / T)
        last = len(self.weights) - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Wt = W.transpose(1, 2)
            z = torch.baddbmm(b.unsqueeze(1), h, Wt)
            dz = torch.bmm(dh, Wt)
            if i < last:
                h = torch.tanh(z)
                dh = (1.0 - h * h) * dz
            else:
                h, dh = z, dz
        raw = self.out_mean + self.out_scale * h
        draw = self.out_scale * dh
        if not self.hard_ic:
            return raw, draw
        s = (t / T).reshape(1, -1, 1)
        x0 = self.x0.unsqueeze(1)
        x = x0 + s * (raw - x0)
        dxdt = (raw - x0) / T + s * draw
        return x, dxdt
