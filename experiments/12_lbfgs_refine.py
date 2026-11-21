"""Exploratory full-window LBFGS refinement of the hard-IC Lorenz pilot.

Starts from the saved 20,000-step Adam model; does not use reference states for
training. The reference is used only by finish_run to evaluate trajectory error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pinn.model import PINN
from src.pinn.residual import lorenz_residual
from src.pinn.runs import finish_run
from src.pinn.train import TrainConfig, weighted_residual_loss
from src.systems.lorenz63 import LorenzParams
from src.utils.reproducibility import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results" / "hard_ic_pilot" / "w0_t2_s0")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "hard_ic_lbfgs_pilot" / "w0_t2_s0")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-collocation", type=int, default=1025)
    args = parser.parse_args()

    job = load_config(args.source / "config.yaml")
    cfg = TrainConfig.from_dict(job["train"])
    if not cfg.hard_ic:
        raise ValueError("source run must use a hard initial condition")
    torch.set_num_threads(cfg.threads)
    model = PINN(cfg.t_final, cfg.out_mean, cfg.out_scale, cfg.hidden, cfg.activation,
                 x0=cfg.x0, hard_ic=True).double()
    model.load_state_dict(torch.load(args.source / "model.pt", map_location="cpu", weights_only=True))
    params = LorenzParams.from_dict(job["params"])
    t = torch.linspace(0, cfg.t_final, args.n_collocation, dtype=torch.float64).unsqueeze(1)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=args.max_iter,
                                  history_size=100, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-12)
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        _, _, residual = lorenz_residual(model, t, params)
        loss = weighted_residual_loss(residual, t, cfg)
        loss.backward()
        calls += 1
        return loss

    start = time.perf_counter()
    initial = closure().item()
    optimizer.step(closure)
    final = closure().item()
    runtime = time.perf_counter() - start
    history = {"iteration": [0, args.max_iter], "loss": [initial, final],
               "loss_ode": [initial, final], "loss_ic": [0.0, 0.0], "lr": [0.8, 0.8],
               "runtime_seconds": runtime, "config": job["train"],
               "refinement": {"optimizer": "LBFGS", "precision": "float64",
                              "max_iter": args.max_iter, "closure_calls": calls,
                              "n_collocation": args.n_collocation,
                              "source": str(args.source), "initial_loss": initial, "final_loss": final}}
    job["run_dir"] = str(args.out)
    job["refinement"] = history["refinement"]
    print(json.dumps(finish_run(job, model, history), indent=2))


if __name__ == "__main__":
    main()
