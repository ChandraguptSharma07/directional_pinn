"""Training a PINN on one Lorenz-63 window with the standard (non-directional) loss (Phase 4).

    L = L_ODE + lambda_IC * L_IC
    L_ODE = mean over collocation times t_i of |r(t_i)|^2
    L_IC  = |x_hat(0) - x0|^2

Collocation times are redrawn at every iteration by stratified sampling: the
window is split into N equal bins and one uniform random time is drawn in each.
The optimiser is Adam with an exponentially decaying learning rate.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from src.pinn.model import PINN
from src.pinn.residual import lorenz_residual
from src.systems.lorenz63 import LorenzParams
from src.utils.seed import set_seed

DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass
class TrainConfig:
    t_final: float
    x0: list[float]
    seed: int = 0
    hidden: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    activation: str = "tanh"
    out_mean: list[float] = field(default_factory=lambda: [0.0, 0.0, 25.0])
    out_scale: list[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])
    iterations: int = 20000
    lr: float = 1e-3
    lr_final: float = 1e-5
    n_collocation: int = 2048
    lambda_ic: float = 10.0
    dtype: str = "float32"
    threads: int = 1
    log_every: int = 100

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        return cls(**values)


def collocation_times(n: int, t_final: float, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """One uniform random time in each of n equal bins of [0, t_final], shape (n, 1)."""
    offsets = torch.rand(n, 1, generator=generator, dtype=dtype)
    return (torch.arange(n, dtype=dtype).unsqueeze(1) + offsets) * (t_final / n)


def train_pinn(cfg: TrainConfig, params: LorenzParams) -> tuple[PINN, dict]:
    """Train a PINN; returns the model and a history (losses, learning rate, run time).

    The run is deterministic for a given config on a given machine: seeds are
    set from cfg.seed and the number of CPU threads is fixed.
    """
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    dtype = DTYPES[cfg.dtype]
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = PINN(cfg.t_final, cfg.out_mean, cfg.out_scale, cfg.hidden, cfg.activation)
    finally:
        torch.set_default_dtype(previous_dtype)
    generator = torch.Generator().manual_seed(cfg.seed)
    x0 = torch.tensor(cfg.x0, dtype=dtype).unsqueeze(0)
    t0 = torch.zeros(1, 1, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gamma = (cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.iterations, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    history = {"iteration": [], "loss": [], "loss_ode": [], "loss_ic": [], "lr": []}
    start = time.perf_counter()
    for iteration in range(1, cfg.iterations + 1):
        t = collocation_times(cfg.n_collocation, cfg.t_final, generator, dtype)
        _, _, residual = lorenz_residual(model, t, params)
        loss_ode = residual.pow(2).sum(dim=1).mean()
        loss_ic = (model(t0) - x0).pow(2).sum()
        loss = loss_ode + cfg.lambda_ic * loss_ic
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if iteration % cfg.log_every == 0 or iteration == cfg.iterations:
            history["iteration"].append(iteration)
            history["loss"].append(loss.item())
            history["loss_ode"].append(loss_ode.item())
            history["loss_ic"].append(loss_ic.item())
            history["lr"].append(scheduler.get_last_lr()[0])
            if not np.isfinite(loss.item()):
                break
    history["runtime_seconds"] = time.perf_counter() - start
    history["config"] = asdict(cfg)
    return model, history


def evaluate(model: PINN, t: np.ndarray, params: LorenzParams, dtype: str = "float64") -> dict[str, np.ndarray]:
    """PINN state, time derivative and residual at times t (shape (N,)), as NumPy arrays.

    By default the trained weights are evaluated in float64, so that rounding in
    the evaluation itself does not add noise to the residual.
    """
    model = copy.deepcopy(model).to(DTYPES[dtype])  # leave the caller's model unchanged
    t_tensor = torch.tensor(np.asarray(t, dtype=float), dtype=DTYPES[dtype]).unsqueeze(1)
    x, dxdt, residual = lorenz_residual(model, t_tensor, params, create_graph=False)
    return {"state": x.detach().numpy(), "derivative": dxdt.detach().numpy(), "residual": residual.detach().numpy()}
