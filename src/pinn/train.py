"""Training a PINN on one Lorenz-63 window with the standard (non-directional) loss (Phase 4).

    L = L_ODE + lambda_IC * L_IC
    L_ODE = mean over collocation times t_i of |r(t_i)|^2
    L_IC  = |x_hat(0) - x0|^2

Collocation times are redrawn at every iteration by stratified sampling: the
interval is split into N equal bins and one uniform random time is drawn in each.
The optimiser is Adam with an exponentially decaying learning rate.

Optional time curriculum: the interval on which collocation times are drawn
grows linearly from [0, curriculum_start] to the full window [0, T] during the
first `curriculum_fraction` of the iterations, and stays [0, T] afterwards. The
loss is the standard one at every iteration, only on a growing interval; at the
end of training it is exactly L on [0, T]. Without it, PINNs on Lorenz windows
of length 2 or more collapse onto the invariant solution x = y = 0 (see
docs/phase4_pinn.md).
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

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
    curriculum_start: float | None = None  # None: no curriculum
    curriculum_fraction: float = 0.6

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        return cls(**values)


def training_interval(cfg: TrainConfig, iteration: int) -> float:
    """Length of the interval [0, length] used for collocation at this iteration (time curriculum)."""
    if cfg.curriculum_start is None:
        return cfg.t_final
    progress = min(1.0, iteration / (cfg.curriculum_fraction * cfg.iterations))
    return cfg.curriculum_start + (cfg.t_final - cfg.curriculum_start) * progress


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
        t = collocation_times(cfg.n_collocation, training_interval(cfg, iteration), generator, dtype)
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


class StackedPINN(nn.Module):
    """Several PINNs of the same architecture, evaluated together with batched matrix products.

    Model m receives its own times t[m] (input shape (M, N, 1), output (M, N, d)).
    The models share no parameters, so training the stack on the sum of their
    losses with Adam (which updates every parameter from its own gradient
    history) is mathematically the same as training each model alone. Batching
    only makes the computation faster.
    """

    def __init__(self, models: list[PINN]) -> None:
        super().__init__()
        linears = [[layer for layer in model.net if isinstance(layer, nn.Linear)] for model in models]
        n_layers = len(linears[0])
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.stack([layers[k].weight.detach() for layers in linears])) for k in range(n_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.stack([layers[k].bias.detach() for layers in linears])) for k in range(n_layers)]
        )
        self.activation = copy.deepcopy(next(layer for layer in models[0].net if not isinstance(layer, nn.Linear)))
        self.t_final = models[0].t_final
        self.register_buffer("out_mean", models[0].out_mean.clone())
        self.register_buffer("out_scale", models[0].out_scale.clone())
        self.template = copy.deepcopy(models[0])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = 2.0 * t / self.t_final - 1.0
        last = len(self.weights) - 1
        for k, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b.unsqueeze(1), h, W.transpose(1, 2))
            if k < last:
                h = self.activation(h)
        return self.out_mean + self.out_scale * h

    def unstack(self, m: int) -> PINN:
        """Model m as an ordinary PINN."""
        model = copy.deepcopy(self.template)
        linears = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        with torch.no_grad():
            for layer, W, b in zip(linears, self.weights, self.biases):
                layer.weight.copy_(W[m])
                layer.bias.copy_(b[m])
        return model


def train_pinns(cfg: TrainConfig, seeds: list[int], params: LorenzParams) -> tuple[list[PINN], list[dict]]:
    """Train one PINN per seed with the settings of cfg (cfg.seed is ignored), all at once.

    Each model gets exactly what ``train_pinn`` would give it: the same
    initialisation (seeded by its seed), its own stream of collocation times,
    and its own loss; only the floating-point rounding of batched products
    differs.
    """
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    dtype = DTYPES[cfg.dtype]
    torch.set_num_threads(cfg.threads)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        models = []
        for seed in seeds:
            set_seed(seed)
            models.append(PINN(cfg.t_final, cfg.out_mean, cfg.out_scale, cfg.hidden, cfg.activation))
        stack = StackedPINN(models)
    finally:
        torch.set_default_dtype(previous_dtype)
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    x0 = torch.tensor(cfg.x0, dtype=dtype).view(1, 1, -1)
    t0 = torch.zeros(len(seeds), 1, 1, dtype=dtype)
    optimizer = torch.optim.Adam(stack.parameters(), lr=cfg.lr)
    gamma = (cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.iterations, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    histories = [{"iteration": [], "loss": [], "loss_ode": [], "loss_ic": [], "lr": []} for _ in seeds]
    start = time.perf_counter()
    for iteration in range(1, cfg.iterations + 1):
        interval = training_interval(cfg, iteration)
        t = torch.stack([collocation_times(cfg.n_collocation, interval, g, dtype) for g in generators])
        _, _, residual = lorenz_residual(stack, t, params)
        loss_ode = residual.pow(2).sum(dim=-1).mean(dim=-1)
        loss_ic = (stack(t0) - x0).pow(2).sum(dim=-1).squeeze(-1)
        losses = loss_ode + cfg.lambda_ic * loss_ic
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        optimizer.step()
        scheduler.step()
        if iteration % cfg.log_every == 0 or iteration == cfg.iterations:
            for m, history in enumerate(histories):
                history["iteration"].append(iteration)
                history["loss"].append(losses[m].item())
                history["loss_ode"].append(loss_ode[m].item())
                history["loss_ic"].append(loss_ic[m].item())
                history["lr"].append(scheduler.get_last_lr()[0])
    runtime = time.perf_counter() - start
    for seed, history in zip(seeds, histories):
        history["runtime_seconds"] = runtime / len(seeds)  # share of the joint run time
        history["config"] = {**asdict(cfg), "seed": seed}
        history["trained_jointly_with_seeds"] = list(seeds)
    return [stack.unstack(m) for m in range(len(seeds))], histories


def evaluate(model: PINN, t: np.ndarray, params: LorenzParams, dtype: str = "float64") -> dict[str, np.ndarray]:
    """PINN state, time derivative and residual at times t (shape (N,)), as NumPy arrays.

    By default the trained weights are evaluated in float64, so that rounding in
    the evaluation itself does not add noise to the residual.
    """
    model = copy.deepcopy(model).to(DTYPES[dtype])  # leave the caller's model unchanged
    t_tensor = torch.tensor(np.asarray(t, dtype=float), dtype=DTYPES[dtype]).unsqueeze(1)
    x, dxdt, residual = lorenz_residual(model, t_tensor, params, create_graph=False)
    return {"state": x.detach().numpy(), "derivative": dxdt.detach().numpy(), "residual": residual.detach().numpy()}
