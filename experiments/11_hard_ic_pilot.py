"""Test exact initial-condition enforcement on the held-out Lorenz pilot window.

Run from the repository root with `python experiments/11_hard_ic_pilot.py`.
The result is exploratory and does not alter the prespecified direction test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pinn.runs import run_job
from src.utils.reproducibility import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--early-weight", type=float, default=0.0)
    parser.add_argument("--time-weighting", choices=["uniform", "turinici", "turinici_code"], default="uniform")
    parser.add_argument("--collocation-scheme", choices=["stratified", "fixed"], default="stratified")
    parser.add_argument("--collocation", type=int)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "hard_ic_pilot" / "w0_t2_s0")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "results" / "interpolation_rebuild" / "reference" / "data" / "window_0.npz")
    args = parser.parse_args()
    config = load_config(ROOT / "config" / "default.yaml")
    x0 = np.load(args.reference)["x0"].tolist()
    train = {**config["pinn"], "t_final": 2.0, "x0": x0, "seed": 0,
             "iterations": args.iterations, "hard_ic": True, "early_weight": args.early_weight,
             "time_weighting": args.time_weighting, "collocation_scheme": args.collocation_scheme,
             "curriculum_start": None if args.collocation_scheme == "fixed" else config["pinn"]["curriculum_start"]}
    if args.collocation is not None:
        train["n_collocation"] = args.collocation
    job = {"run_dir": str(args.out), "window": 0, "reference_file": str(args.reference),
           "params": config["lorenz"]["params"], "train": train}
    print(json.dumps(run_job(job), indent=2))


if __name__ == "__main__":
    main()
