"""Training one batch of PINNs under one residual weighting (docs/plan.md, sections 4-6)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from mw import weights as wt
from mw.model import BatchedPINN
from mw.systems import System

DTYPES = {"float32": torch.float32, "float64": torch.float64}
TANGENT_SELF = ("S-self", "M-self", "M-final-self")
TANGENT_ORACLE = ("S-orc", "M-orc", "M-final-orc")


@dataclass
class Settings:
    t_final: float
    hidden: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    n_grid: int = 512
    iterations: int = 20000
    lr: float = 1e-3
    lr_final: float = 1e-5
    weight_every: int = 10  # self-consistent tangent weights are recomputed every this many steps
    causal_eps: float = 1.0
    dtype: str = "float32"
    device: str = "cpu"
    log_every: int = 500
    n_eval: int = 2001
    verbose: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        return cls(**d)


def grid(T: float, n: int) -> np.ndarray:
    return np.linspace(0.0, T, n)


def train_batch(system: System, method: str, x0: np.ndarray, init_seeds: list[int], out_mean, out_scale,
                settings: Settings, ref_grid: np.ndarray | None = None, ref_eval: np.ndarray | None = None) -> dict:
    """Train len(x0) independent networks with one weighting method.

    ref_grid (batch, n_grid, n) is the reference on the collocation grid. Oracle
    methods use it for their weights. All methods use it only to log
    trajectory error, never in the loss. ref_eval (batch, n_eval, n) is used
    for the final metrics.
    """
    if method not in wt.METHODS:
        raise ValueError(f"unknown method {method!r}")
    s = settings
    dtype = DTYPES[s.dtype]
    device = torch.device(s.device)
    T, N = float(s.t_final), int(s.n_grid)
    h = T / (N - 1)
    t = torch.tensor(grid(T, N), dtype=dtype, device=device)
    model = BatchedPINN(x0, T, out_mean, out_scale, init_seeds, s.hidden, dtype=dtype).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=s.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(s.lr_final / s.lr) ** (1.0 / max(s.iterations, 1)))

    ref_grid_t = None if ref_grid is None else torch.tensor(ref_grid, dtype=torch.float64, device=device)
    kind, W = None, None
    if method in TANGENT_ORACLE:
        if ref_grid_t is None:
            raise ValueError("oracle weights need the reference on the grid")
        kind, W = wt.tangent_weight(method, ref_grid_t, system.jac, h)
        W = W.to(dtype)

    hist = {"iteration": [], "loss_weighted": [], "residual_rms": [], "grid_error_rms": [], "weight_time_profile": []}
    start = time.perf_counter()
    for it in range(s.iterations + 1):
        x, dx = model(t)
        r = dx - system.rhs(x)
        if method in TANGENT_SELF and it % s.weight_every == 0:
            kind, W = wt.tangent_weight(method, x, system.jac, h)
            W = W.to(dtype)
        elif method == "TUR":
            kind, W = "scalar", wt.turinici(x.detach().double(), system.rhs, h).to(dtype)
        elif method == "causal":
            kind, W = "scalar", wt.causal(r.detach().double(), h, s.causal_eps).to(dtype)
        per_net = wt.weighted_loss(r, kind, W)
        if it % s.log_every == 0 or it == s.iterations:
            with torch.no_grad():
                hist["iteration"].append(it)
                hist["loss_weighted"].append(per_net.detach().cpu().double().numpy().tolist())
                hist["residual_rms"].append(r.detach().square().sum(-1).mean(1).sqrt().cpu().double().numpy().tolist())
                if ref_grid_t is not None:
                    e = (x.detach().double() - ref_grid_t).square().sum(-1).mean(1).sqrt()
                    hist["grid_error_rms"].append(e.cpu().numpy().tolist())
                if W is not None:
                    prof = W if kind == "scalar" else W.diagonal(dim1=-2, dim2=-1).sum(-1) / W.shape[-1]
                    hist["weight_time_profile"].append(prof[:, :: max(1, N // 64)].detach().cpu().double().numpy().tolist())
                if s.verbose:
                    ge = hist["grid_error_rms"][-1] if hist["grid_error_rms"] else [float("nan")]
                    print(f"it {it:6d}  {time.perf_counter() - start:7.1f} s  median residual rms "
                          f"{np.median(hist['residual_rms'][-1]):.3e}  median grid error rms {np.median(ge):.3e}",
                          flush=True)
        if it == s.iterations:
            break
        opt.zero_grad(set_to_none=True)
        per_net.sum().backward()
        opt.step()
        sched.step()
    runtime = time.perf_counter() - start

    # Final evaluation in float64 on the evaluation grid.
    model64 = model.to(torch.float64)
    te = torch.tensor(grid(T, s.n_eval), dtype=torch.float64, device=device)
    with torch.no_grad():
        xe, dxe = model64(te)
        re = dxe - system.rhs(xe)
    out = {"x_eval": xe.cpu().numpy(), "r_eval": re.cpu().numpy(), "history": hist, "runtime_seconds": runtime,
           "settings": asdict(s), "method": method}
    if ref_eval is not None:
        out["metrics"] = trajectory_metrics(out["x_eval"], ref_eval, out["r_eval"])
    return out


def trajectory_metrics(x_hat: np.ndarray, x_ref: np.ndarray, r: np.ndarray | None = None,
                       components: slice | None = None) -> dict:
    """Per-network error summaries on the evaluation grid (lists, one entry per network)."""
    sl = components if components is not None else slice(None)
    diff = x_hat[..., sl] - x_ref[..., sl]
    err = np.sqrt(np.sum(diff**2, axis=-1))
    finite = np.all(np.isfinite(err), axis=1)
    err = np.where(np.isfinite(err), err, np.inf)
    ref_rms = np.sqrt(np.mean(np.sum(x_ref[..., sl] ** 2, axis=-1), axis=1))
    above = err > 1.0
    first_above = np.where(above.any(axis=1), above.argmax(axis=1), -1)
    m = {
        "E_rms": np.sqrt(np.mean(err**2, axis=1)).tolist(),
        "E_max": err.max(axis=1).tolist(),
        "E_T": err[:, -1].tolist(),
        "ref_rms": ref_rms.tolist(),
        "finite": finite.tolist(),
        "success_max_below_1": (err.max(axis=1) < 1.0).tolist(),
        "first_index_error_above_1": first_above.tolist(),
    }
    if r is not None:
        m["residual_rms"] = np.sqrt(np.mean(np.sum(r**2, axis=-1), axis=1)).tolist()
    return m
