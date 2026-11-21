"""Check continuity at joins of a saved piecewise PINN run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pinn.model import PINN
from src.pinn.residual import state_and_time_derivative


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    starts, ends, start_slopes, end_slopes = [], [], [], []
    for k in range(metrics["segments"]):
        history = json.loads((args.run / f"segment_{k:02d}_history.json").read_text())
        cfg = history["config"]
        model = PINN(cfg["t_final"], cfg["out_mean"], cfg["out_scale"], cfg["hidden"],
                     cfg["activation"], x0=cfg["x0"], hard_ic=True).double()
        model.load_state_dict(torch.load(args.run / f"segment_{k:02d}.pt",
                                         map_location="cpu", weights_only=True))
        for point, states, slopes in ((0.0, starts, start_slopes),
                                      (cfg["t_final"], ends, end_slopes)):
            state, slope = state_and_time_derivative(
                model, torch.tensor([[point]], dtype=torch.float64), create_graph=False)
            states.append(state.detach().numpy()[0])
            slopes.append(slope.detach().numpy()[0])
    jumps = [float(np.linalg.norm(ends[k] - starts[k + 1])) for k in range(len(ends) - 1)]
    slope_jumps = [float(np.linalg.norm(end_slopes[k] - start_slopes[k + 1]))
                   for k in range(len(ends) - 1)]
    print(json.dumps({"run": str(args.run), "max_join_jump": max(jumps) if jumps else 0.0,
                      "join_jumps": jumps,
                      "max_derivative_jump": max(slope_jumps) if slope_jumps else 0.0,
                      "derivative_jumps": slope_jumps}, indent=2))


if __name__ == "__main__":
    main()
