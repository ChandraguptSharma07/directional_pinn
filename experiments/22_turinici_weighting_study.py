"""Matched seed-0 fixed-grid sequential comparison across Lorenz windows 1--8.

Each job calls experiment 13 with the same reference, architecture, optimizer,
segment count, collocation grid, and seed. Only the residual time weights differ.
The output summary is updated after every completed job so an interrupted run
can resume without discarding completed trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(window: int, weighting: str, out_root: Path, reference_root: Path) -> dict:
    out = out_root / f"w{window}_t2_piecewise_{weighting}_fixed"
    metrics_file = out / "metrics.json"
    if not metrics_file.exists():
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "experiments" / "13_piecewise_pinn_pilot.py"),
                   "--window", str(window), "--seed", "0", "--collocation-scheme", "fixed",
                   "--time-weighting", weighting, "--reference", str(reference_root / f"window_{window}.npz"),
                   "--out", str(out)]
        with (out / "training.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    with metrics_file.open() as file:
        metrics = json.load(file)
    return {"window": window, "weighting": weighting, "median_error": metrics["median_error"],
            "max_error": metrics["max_error"], "final_error": metrics["final_error"],
            "fraction_error_below_1": metrics["fraction_error_below_1"],
            "runtime_seconds": metrics["runtime_seconds"], "metrics_file": str(metrics_file)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=["uniform", "turinici", "turinici_code"],
                        default=["uniform", "turinici"])
    parser.add_argument("--summary-name", default="sequential_seed0_summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "adaptive_time_weighting")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "results" / "interpolation_rebuild" / "reference" / "data")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(window, weighting) for window in args.windows for weighting in args.methods]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, w, method, args.out, args.reference): (w, method) for w, method in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: (item["window"], item["weighting"]))
            (args.out / args.summary_name).write_text(json.dumps(rows, indent=2) + "\n")
            print(f"{len(rows)}/{len(jobs)} window={row['window']} {row['weighting']} "
                  f"median={row['median_error']:.5f} max={row['max_error']:.5f}", flush=True)


if __name__ == "__main__":
    main()
