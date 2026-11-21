"""Package saved piecewise PINNs for the existing directional analysis code."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.reproducibility import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results" / "piecewise_pinn_pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "piecewise_seed0_analysis_runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    summaries = []
    for window in range(1, 9):
        for seed in args.seeds:
            source = args.source / f"w{window}_t2_s{seed}"
            metrics = json.loads((source / "metrics.json").read_text())
            trajectory = np.load(source / "trajectory.npz")
            residuals = np.load(source / "residuals.npz")
            metrics["finite"] = bool(np.all(np.isfinite(trajectory["error"]))
                                     and np.all(np.isfinite(residuals["residual"])))
            target = args.out / "runs" / f"w{window}_s{seed}"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "trajectory.npz", target / "trajectory.npz")
            shutil.copy2(source / "residuals.npz", target / "residuals.npz")
            save_json(metrics, target / "metrics.json")
            summaries.append(metrics)
    save_json({"source": str(args.source), "seeds": args.seeds, "runs": summaries}, args.out / "summary.json")
    print(f"Prepared {len(summaries)} runs in {args.out}")


if __name__ == "__main__":
    main()
