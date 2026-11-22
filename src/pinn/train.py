"""Training a PINN on one window with the standard (non-directional) loss (Phase 4).

The ODE is Lorenz-63 by default; any system with ``rhs_torch`` can be passed instead
(see ``system_rhs``).

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
from typing import Callable

import numpy as np
import torch
from torch import nn

from src.pinn.model import PINN
from src.pinn.residual import lorenz_rhs_torch, ode_residual
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
    hard_ic: bool = False
    early_weight: float = 0.0  # normalized linear emphasis on earlier residuals
    time_weighting: str = "uniform"  # "turinici": Eq. 13; "turinici_code": literal Lorenz-script time scale
    collocation_scheme: str = "stratified"  # "stratified" or "fixed" regular grid
    dtype: str = "float32"
    threads: int = 1
    log_every: int = 100
    curriculum_start: float | None = None  # None: no curriculum
    curriculum_fraction: float = 0.6
    device: str = "cpu"  # "cuda" to train on a GPU (stacked training only)

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        return cls(**values)


def system_rhs(params) -> Callable[[torch.Tensor], torch.Tensor]:
    """Right-hand side f used in the residual r = dx_hat/dt - f(x_hat).

    ``params`` is either LorenzParams (Lorenz-63) or any system object with a
    ``rhs_torch(x)`` method, such as those in src/systems/dial_systems.py.
    """
    if isinstance(params, LorenzParams):
        return lambda x: lorenz_rhs_torch(x, params)
    if not hasattr(params, "rhs_torch"):
        raise TypeError("params must be LorenzParams or provide rhs_torch(x)")
    return params.rhs_torch


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


def turinici_time_weights(states: torch.Tensor, t_final: float, params: LorenzParams,
                          code_time_scale: bool = False) -> torch.Tensor:
    """Discrete Eq. (13) using the paper's on-the-fly state/zero growth proxy.

    The final dimension is state; the preceding one is an ordered fixed time grid.
    Weights are detached so that training differentiates only the residual loss,
    just as the paper updates its weights outside the optimizer gradient.
    """
    with torch.no_grad():
        if states.shape[-2] < 2:
            raise ValueError("adaptive weighting needs at least two time points")
        rate = (lorenz_rhs_torch(states, params) * states).sum(dim=-1) / (states.square().sum(dim=-1) + 1e-10)
        dt = (1.0 if code_time_scale else t_final) / (states.shape[-2] - 1)
        log_weight = -torch.cumsum(rate, dim=-1) * dt
        return torch.softmax(log_weight, dim=-1)


def weighted_residual_loss(residual: torch.Tensor, t: torch.Tensor, cfg: TrainConfig,
                           time_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Mean squared residual per model, with optional extra weight near t=0."""
    if cfg.early_weight < 0:
        raise ValueError("early_weight must be nonnegative")
    if time_weights is not None:
        if cfg.early_weight != 0:
            raise ValueError("adaptive and fixed early weighting cannot be combined")
        return (residual.square().sum(dim=-1) * time_weights).sum(dim=-1)
    weight = 1.0 + cfg.early_weight * (1.0 - t.squeeze(-1) / cfg.t_final)
    return (residual.square().sum(dim=-1) * weight).sum(dim=-1) / weight.sum(dim=-1)


def train_pinn(cfg: TrainConfig, params: LorenzParams) -> tuple[PINN, dict]:
    """Train a PINN; returns the model and a history (losses, learning rate, run time).

    The run is deterministic for a given config on a given machine: seeds are
    set from cfg.seed and the number of CPU threads is fixed.
    """
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if cfg.time_weighting not in ("uniform", "turinici", "turinici_code"):
        raise ValueError("time_weighting must be uniform, turinici, or turinici_code")
    if cfg.collocation_scheme not in ("stratified", "fixed"):
        raise ValueError("collocation_scheme must be stratified or fixed")
    if cfg.time_weighting != "uniform" and (cfg.collocation_scheme != "fixed" or cfg.curriculum_start is not None):
        raise ValueError("Turinici weighting requires a fixed full-window grid")
    if cfg.time_weighting != "uniform" and not isinstance(params, LorenzParams):
        raise ValueError("Turinici weighting is implemented for Lorenz-63 only")
    rhs = system_rhs(params)
    dtype = DTYPES[cfg.dtype]
    torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = PINN(cfg.t_final, cfg.out_mean, cfg.out_scale, cfg.hidden, cfg.activation,
                     x0=cfg.x0, hard_ic=cfg.hard_ic)
    finally:
        torch.set_default_dtype(previous_dtype)
    generator = torch.Generator().manual_seed(cfg.seed)
    x0 = torch.tensor(cfg.x0, dtype=dtype).unsqueeze(0)
    t0 = torch.zeros(1, 1, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gamma = (cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.iterations, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    history = {"iteration": [], "loss": [], "loss_ode": [], "loss_ic": [], "lr": [], "weight_contrast": []}
    fixed_t = torch.linspace(0.0, cfg.t_final, cfg.n_collocation, dtype=dtype).unsqueeze(1) if cfg.collocation_scheme == "fixed" else None
    time_weights = torch.full((cfg.n_collocation,), 1.0 / cfg.n_collocation, dtype=dtype) if cfg.time_weighting != "uniform" else None
    start = time.perf_counter()
    for iteration in range(1, cfg.iterations + 1):
        t = fixed_t if fixed_t is not None else collocation_times(cfg.n_collocation, training_interval(cfg, iteration), generator, dtype)
        _, _, residual = ode_residual(model, t, rhs)
        loss_ode = weighted_residual_loss(residual, t, cfg, time_weights)
        loss_ic = (model(t0) - x0).pow(2).sum()
        loss = loss_ode + cfg.lambda_ic * loss_ic
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if time_weights is not None:
            with torch.no_grad():
                time_weights = turinici_time_weights(model(t), cfg.t_final, params,
                                                      code_time_scale=cfg.time_weighting == "turinici_code")
        if iteration % cfg.log_every == 0 or iteration == cfg.iterations:
            history["iteration"].append(iteration)
            history["loss"].append(loss.item())
            history["loss_ode"].append(loss_ode.item())
            history["loss_ic"].append(loss_ic.item())
            history["lr"].append(scheduler.get_last_lr()[0])
            history["weight_contrast"].append(float((time_weights.max() / time_weights.min()).item()) if time_weights is not None else 1.0)
            if not np.isfinite(loss.item()):
                break
    history["runtime_seconds"] = time.perf_counter() - start
    history["config"] = asdict(cfg)
    if time_weights is not None:
        history["final_time_weights"] = time_weights.tolist()
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
        self.hard_ic = models[0].hard_ic
        self.register_buffer("out_mean", models[0].out_mean.clone())
        self.register_buffer("out_scale", models[0].out_scale.clone())
        self.register_buffer("x0", torch.stack([model.x0.clone() for model in models]).unsqueeze(1))
        self.template = copy.deepcopy(models[0])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = 2.0 * t / self.t_final - 1.0
        last = len(self.weights) - 1
        for k, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b.unsqueeze(1), h, W.transpose(1, 2))
            if k < last:
                h = self.activation(h)
        raw = self.out_mean + self.out_scale * h
        return self.x0 + (t / self.t_final) * (raw - self.x0) if self.hard_ic else raw

    def unstack(self, m: int) -> PINN:
        """Model m as an ordinary PINN on the CPU."""
        model = copy.deepcopy(self.template).cpu()
        linears = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        with torch.no_grad():
            model.x0.copy_(self.x0[m, 0].cpu())
            for layer, W, b in zip(linears, self.weights, self.biases):
                layer.weight.copy_(W[m].cpu())
                layer.bias.copy_(b[m].cpu())
        return model


def train_pinns(
    cfg: TrainConfig, seeds: list[int], params: LorenzParams, x0s: list[list[float]] | None = None
) -> tuple[list[PINN], list[dict]]:
    """Train one PINN per seed with the settings of cfg (cfg.seed is ignored), all at once.

    Each model gets exactly what ``train_pinn`` would give it: the same
    initialisation (seeded by its seed), its own stream of collocation times
    (generated on the CPU, so they do not depend on the device), and its own
    loss; only the floating-point rounding of batched products differs.
    Models may have different initial conditions (`x0s`, one per seed;
    default: cfg.x0 for all), so that runs from several windows can share one stack.
    """
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if cfg.time_weighting != "uniform" or cfg.collocation_scheme != "stratified":
        raise ValueError("adaptive or fixed-grid training currently supports train_pinn only")
    rhs = system_rhs(params)
    dtype = DTYPES[cfg.dtype]
    torch.set_num_threads(cfg.threads)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        models = []
        initial_conditions = x0s if x0s is not None else [cfg.x0] * len(seeds)
        for seed, initial_condition in zip(seeds, initial_conditions):
            set_seed(seed)
            models.append(PINN(cfg.t_final, cfg.out_mean, cfg.out_scale, cfg.hidden, cfg.activation,
                               x0=initial_condition, hard_ic=cfg.hard_ic))
        stack = StackedPINN(models)
    finally:
        torch.set_default_dtype(previous_dtype)
    device = torch.device(cfg.device)
    stack = stack.to(device)
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    x0 = torch.tensor(x0s if x0s is not None else [cfg.x0] * len(seeds), dtype=dtype, device=device).unsqueeze(1)
    t0 = torch.zeros(len(seeds), 1, 1, dtype=dtype, device=device)
    optimizer = torch.optim.Adam(stack.parameters(), lr=cfg.lr)
    gamma = (cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.iterations, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    histories = [{"iteration": [], "loss": [], "loss_ode": [], "loss_ic": [], "lr": []} for _ in seeds]
    start = time.perf_counter()
    for iteration in range(1, cfg.iterations + 1):
        interval = training_interval(cfg, iteration)
        t = torch.stack([collocation_times(cfg.n_collocation, interval, g, dtype) for g in generators]).to(device)
        _, _, residual = ode_residual(stack, t, rhs)
        loss_ode = weighted_residual_loss(residual, t, cfg)
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
    for m, (seed, history) in enumerate(zip(seeds, histories)):
        history["runtime_seconds"] = runtime / len(seeds)  # share of the joint run time
        history["config"] = {**asdict(cfg), "seed": seed, "x0": x0[m, 0].tolist()}
        history["trained_jointly_with_seeds"] = list(seeds)
        if device.type == "cuda":
            history["device_name"] = torch.cuda.get_device_name(device)
    return [stack.unstack(m) for m in range(len(seeds))], histories


def evaluate(model: PINN, t: np.ndarray, params: LorenzParams, dtype: str = "float64") -> dict[str, np.ndarray]:
    """PINN state, time derivative and residual at times t (shape (N,)), as NumPy arrays.

    By default the trained weights are evaluated in float64, so that rounding in
    the evaluation itself does not add noise to the residual.
    """
    model = copy.deepcopy(model).to(DTYPES[dtype])  # leave the caller's model unchanged
    t_tensor = torch.tensor(np.asarray(t, dtype=float), dtype=DTYPES[dtype]).unsqueeze(1)
    x, dxdt, residual = ode_residual(model, t_tensor, system_rhs(params), create_graph=False)
    return {"state": x.detach().numpy(), "derivative": dxdt.detach().numpy(), "residual": residual.detach().numpy()}
