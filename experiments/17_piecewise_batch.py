"""Run the fixed piecewise PINN settings over the analysis windows and seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "piecewise_pinn_pilot")
    args = parser.parse_args()

    def run(window: int, seed: int) -> tuple[int, int, dict]:
        target = args.out / f"w{window}_t2_s{seed}"
        metrics_file = target / "metrics.json"
        if not metrics_file.exists():
            command = [sys.executable, str(ROOT / "experiments" / "13_piecewise_pinn_pilot.py"),
                       "--window", str(window), "--seed", str(seed), "--out", str(target)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"window {window}, seed {seed}: {result.stdout}\n{result.stderr}")
        return window, seed, json.loads(metrics_file.read_text())

    jobs = [(window, seed) for window in args.windows for seed in args.seeds]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, window, seed): (window, seed) for window, seed in jobs}
        for future in as_completed(futures):
            window, seed, metrics = future.result()
            print(f"window {window}, seed {seed}: median {metrics['median_error']:.4g}, "
                  f"max {metrics['max_error']:.4g}, below 1 {metrics['fraction_error_below_1']:.1%}", flush=True)

    all_metrics = []
    for window in range(1, 9):
        for seed in range(5):
            all_metrics.append(json.loads((args.out / f"w{window}_t2_s{seed}" / "metrics.json").read_text()))
    summary = {"experiment": "17_piecewise_batch", "n_runs": len(all_metrics),
               "median_of_run_medians": float(np.median([m["median_error"] for m in all_metrics])),
               "median_of_run_maxima": float(np.median([m["max_error"] for m in all_metrics])),
               "runs_below_1_throughout": sum(m["fraction_error_below_1"] == 1 for m in all_metrics),
               "runs_below_1_at_least_90_percent": sum(m["fraction_error_below_1"] >= 0.9 for m in all_metrics),
               "runs": all_metrics}
    (args.out / "validation_40runs_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
