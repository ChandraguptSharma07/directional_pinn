"""Phases 4-7: train the baseline PINN on every analysis window with several seeds.

Run from the repository root (after experiments/03 and 05):

    python experiments/06_pinn_runs.py                       # float32 runs (main analysis)
    python experiments/06_pinn_runs.py --set float64_check   # float64 precision check

Every (window, seed) pair is one run, saved under results/<set>/runs/w<k>_s<seed>/
(see src/pinn/runs.py for the files). Finished runs are skipped, so an
interrupted call can simply be repeated. No run is dropped from the summary,
whatever its outcome.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/06_pinn_runs.py` import src

import json

import numpy as np

from src.pinn.runs import run_job
from src.utils.reproducibility import load_config, save_json, write_run_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the baseline PINN on all analysis windows and seeds.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--reference", type=Path, default=REPO_ROOT / "results" / "lorenz_reference")
    parser.add_argument("--set", default="pinn_runs", help="config section describing the set of runs")
    parser.add_argument("--out", type=Path, default=None, help="default: results/<set>")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_config(args.config)
    run_set = config[args.set]
    out = args.out or REPO_ROOT / "results" / args.set
    jobs = []
    for window in run_set["windows"]:
        reference_file = args.reference / "data" / f"window_{window}.npz"
        x0 = np.load(reference_file)["x0"].tolist()
        for seed in run_set["seeds"]:
            run_dir = out / "runs" / f"w{window}_s{seed}"
            if (run_dir / "metrics.json").exists():
                continue
            train = {**config["pinn"], **run_set.get("overrides", {}), "x0": x0, "seed": int(seed)}
            jobs.append({"run_dir": str(run_dir), "window": int(window), "reference_file": str(reference_file),
                         "params": config["lorenz"]["params"], "train": train})
    print(f"{len(jobs)} runs to train")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for summary in pool.map(run_job, jobs):
            print(f"window {summary['window']} seed {summary['seed']}: median error {summary['median_error']:.3g},"
                  f" max {summary['max_error']:.3g}, {summary['runtime_seconds']:.0f}s", flush=True)

    summaries = [json.loads((d / "metrics.json").read_text()) for d in sorted((out / "runs").iterdir())]
    save_json({"set": args.set, "runs": summaries}, out / "summary.json")
    write_run_record(out, {key: config[key] for key in ("lorenz", "pinn", args.set)}, REPO_ROOT,
                     time.perf_counter() - start)
    print(f"{len(summaries)} runs summarised in {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
