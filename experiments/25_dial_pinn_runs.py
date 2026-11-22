"""Dial study, step 2: train the PINNs of docs/dial_study_plan.md (section 5 and its amendment).

The settings are those of the plan:
* network: four tanh layers of 64 units;
* training: Adam with the learning rate decaying from 1e-3 to 1e-5, 1024
  stratified collocation points, soft initial condition with lambda_IC = 100,
  float32, T = 2;
* output scaling: mean and standard deviation of the setting's reference
  states (from experiments/24_dial_reference.py).

    python experiments/25_dial_pinn_runs.py --pilot
        window 0, seed 0, at the stiffest setting of each family, then the
        accuracy check of the pilot rule (no direction statistics)
    python experiments/25_dial_pinn_runs.py --iterations 20000 --device cuda
        all 8 settings x windows 1-8 x seeds 0-4 (320 runs), one stacked model per
        setting. Finished runs (metrics.json present) are skipped, so the command
        can be restarted after an interruption.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pinn.runs import run_job, run_jobs_together  # noqa: E402
from src.pinn.train import TrainConfig  # noqa: E402
from src.systems.dial_systems import DavisSkodje, StiffSaddle, dial_settings  # noqa: E402
from src.utils.reproducibility import save_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "results/dial_study/reference"
SEEDS = range(5)
ANALYSIS_WINDOWS = range(1, 9)
PILOT_SETTINGS = (StiffSaddle(64.0), DavisSkodje(66.0))
PILOT_TOLERANCE = 0.05  # max error <= 5% of the max reference state norm


def make_job(system, window: int, seed: int, iterations: int, device: str, out: Path) -> dict:
    scaling = json.loads((REFERENCE / "metrics.json").read_text())["settings"][system.name]
    cfg = TrainConfig(
        t_final=2.0, x0=system.initial_state(window).tolist(), seed=seed, hidden=[64, 64, 64, 64],
        activation="tanh", out_mean=scaling["out_mean"], out_scale=scaling["out_scale"],
        iterations=iterations, lr=1e-3, lr_final=1e-5, n_collocation=1024, lambda_ic=100.0,
        hard_ic=False, dtype="float32", threads=1, log_every=100, curriculum_start=None, device=device,
    )
    return {
        "run_dir": str(out / system.name / "runs" / f"w{window}_s{seed}"),
        "window": window,
        "reference_file": str(REFERENCE / system.name / "data" / f"window_{window}.npz"),
        "system": system.to_dict(),
        "train": asdict(cfg),
    }


def pilot(args) -> None:
    jobs = [make_job(s, 0, 0, args.iterations, "cpu", args.out / "pilot") for s in PILOT_SETTINGS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        summaries = list(pool.map(run_job, jobs))
    verdict = {}
    for system, summary in zip(PILOT_SETTINGS, summaries):
        relative = summary["max_error"] / summary["max_reference_norm"]
        verdict[system.name] = {"max_error": summary["max_error"], "max_reference_norm": summary["max_reference_norm"],
                                "relative_max_error": relative, "passes": bool(relative <= PILOT_TOLERANCE),
                                "median_error": summary["median_error"], "runtime_seconds": summary["runtime_seconds"]}
        print(f"{system.name}: max error {summary['max_error']:.4g} = {relative:.2%} of max state norm "
              f"({'pass' if relative <= PILOT_TOLERANCE else 'FAIL'}), median error {summary['median_error']:.3g}")
    verdict["iterations"] = args.iterations
    verdict["all_pass"] = all(v["passes"] for k, v in verdict.items() if isinstance(v, dict))
    save_json(verdict, args.out / "pilot" / f"pilot_{args.iterations}.json")


def main_runs(args) -> None:
    start = time.perf_counter()
    for system in dial_settings():
        jobs = [make_job(system, w, s, args.iterations, args.device, args.out) for w in ANALYSIS_WINDOWS for s in SEEDS]
        todo = [job for job in jobs if not (Path(job["run_dir"]) / "metrics.json").exists()]
        print(f"{system.name}: {len(todo)}/{len(jobs)} runs to train", flush=True)
        if todo:
            t0 = time.perf_counter()
            summaries = run_jobs_together(todo)
            finite = sum(s["finite"] for s in summaries)
            print(f"{system.name}: done in {time.perf_counter() - t0:.0f} s, {finite}/{len(summaries)} finite", flush=True)
    rows = []
    for system in dial_settings():
        for run_dir in sorted((args.out / system.name / "runs").glob("w*_s*")):
            summary = json.loads((run_dir / "metrics.json").read_text())
            rows.append({"setting": system.name, "dial": system.dial, **{k: summary[k] for k in (
                "window", "seed", "finite", "median_error", "max_error", "max_reference_norm", "median_residual",
                "initial_error")}})
    save_json({"runs": rows, "n_runs": len(rows), "total_seconds": time.perf_counter() - start},
              args.out / "pinn_summary.json")
    print(f"{len(rows)} runs summarised", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/dial_study")
    args = parser.parse_args()
    pilot(args) if args.pilot else main_runs(args)


if __name__ == "__main__":
    main()
