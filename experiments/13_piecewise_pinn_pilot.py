"""Train consecutive hard-IC PINNs on a Lorenz pilot window.

Only the state at t=0 is read from the reference for training. Each following
segment starts from the previous PINN's predicted endpoint. Reference states
along the window are used only for evaluation after training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pinn.train import TrainConfig, evaluate, train_pinn
from src.systems.lorenz63 import LorenzParams
from src.utils.reproducibility import load_config, save_json, write_run_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--collocation", type=int, default=256)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.segments <= 0 or args.iterations <= 0 or args.collocation <= 0:
        raise ValueError("segments, iterations, and collocation must be positive")
    reference_file = args.reference or (ROOT / "results" / "interpolation_rebuild" / "reference" / "data"
                                         / f"window_{args.window}.npz")
    out = args.out or (ROOT / "results" / "piecewise_pinn_pilot" / f"w{args.window}_t2_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "config" / "default.yaml")
    reference = np.load(reference_file)
    full_t = reference["t"][reference["t"] <= 2.0 + 1e-9]
    full_ref = reference["states"][:len(full_t)]
    x0 = reference["x0"].tolist()
    params = LorenzParams.from_dict(config["lorenz"]["params"])
    length = 2.0 / args.segments
    state_parts, residual_parts, time_parts, segment_metrics = [], [], [], []
    start_time = time.perf_counter()

    for k in range(args.segments):
        left, right = k * length, (k + 1) * length
        train = {**config["pinn"], "t_final": length, "x0": x0, "seed": args.seed,
                 "iterations": args.iterations, "n_collocation": args.collocation,
                 "hard_ic": True, "early_weight": 0.0, "curriculum_start": None}
        cfg = TrainConfig.from_dict(train)
        model, history = train_pinn(cfg, params)
        mask = (full_t >= left - 1e-9) & (full_t <= right + 1e-9)
        global_t = full_t[mask]
        values = evaluate(model, global_t - left, params)
        x0 = values["state"][-1].tolist()
        sl = slice(None) if k == 0 else slice(1, None)
        time_parts.append(global_t[sl])
        state_parts.append(values["state"][sl])
        residual_parts.append(values["residual"][sl])
        torch.save(model.state_dict(), out / f"segment_{k:02d}.pt")
        save_json(history, out / f"segment_{k:02d}_history.json")
        segment_metrics.append({"segment": k, "t_start": left, "t_end": right,
                                "loss": history["loss"][-1],
                                "runtime_seconds": history["runtime_seconds"]})
        print(f"segment {k+1}/{args.segments}: loss {history['loss'][-1]:.4g}", flush=True)

    t = np.concatenate(time_parts)
    state = np.concatenate(state_parts)
    residual = np.concatenate(residual_parts)
    if len(t) != len(full_t) or not np.allclose(t, full_t):
        raise RuntimeError("piecewise evaluation did not cover the reference grid exactly")
    error = state - full_ref
    size = np.linalg.norm(error, axis=1)
    residual_size = np.linalg.norm(residual, axis=1)
    above = np.flatnonzero(size > 1.0)
    metrics = {"experiment": "13_piecewise_pinn_pilot", "window": args.window, "seed": args.seed,
               "t_final": 2.0, "segments": args.segments, "segment_length": length,
               "iterations_per_segment": args.iterations, "collocation_per_segment": args.collocation,
               "initial_error": float(size[0]), "median_error": float(np.median(size)),
               "max_error": float(size.max()), "final_error": float(size[-1]),
               "fraction_error_below_1": float(np.mean(size < 1)),
               "first_error_above_1": float(t[above[0]]) if len(above) else None,
               "median_residual": float(np.median(residual_size)),
               "runtime_seconds": time.perf_counter() - start_time,
               "segment_metrics": segment_metrics}
    np.savez(out / "trajectory.npz", t=t, reference=full_ref, pinn=state, error=error)
    np.savez(out / "residuals.npz", t=t, residual=residual)
    save_json(metrics, out / "metrics.json")
    write_run_record(out, {"reference_file": str(reference_file),
                                    "params": config["lorenz"]["params"],
                                    "train": train, "segments": args.segments}, ROOT,
                     metrics["runtime_seconds"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
