"""Run every job file in a directory with a pool of worker processes.

Finished jobs (summary.json present) are skipped, so the pool can be restarted
after an interruption. Each job runs in its own process with its own thread
count. Its output goes to <out_dir>/log.txt.

    python experiments/pool.py jobs/gate0 --workers 14 --nice 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_one(job_file: Path, nice: int, device: str | None, threads: int | None = None) -> tuple[str, int, float]:
    job = json.loads(job_file.read_text())
    if threads:
        job["threads"] = threads
    out = ROOT / job["out_dir"]
    if (out / "summary.json").exists():
        return job["out_dir"], 0, 0.0
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(job.get("threads", 1)), "MKL_NUM_THREADS": str(job.get("threads", 1))}
    cmd = [sys.executable, str(ROOT / "experiments" / "run_job.py"), str(job_file)]
    if threads:
        cmd += ["--threads", str(threads)]
    if device:
        cmd += ["--device", device]
    if nice:
        cmd = ["nice", "-n", str(nice), *cmd]
    start = time.perf_counter()
    with (out / "log.txt").open("a") as log:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return job["out_dir"], code, time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path, nargs="+", help="job directories or job files")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None, help="override the thread count of every job")
    args = ap.parse_args()
    files: list[Path] = []
    for p in args.jobs:
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    # Longest jobs first (larger batch, longer window) to shorten the tail.
    def cost(f: Path) -> float:
        j = json.loads(f.read_text())
        return len(j["ics"]) * len(j["seeds"]) * j["settings"].get("iterations", 20000) * (2 if "self" in j["method"] else 1)
    files.sort(key=cost, reverse=True)
    print(f"{len(files)} job files, {args.workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, f, args.nice, args.device, args.threads): f for f in files}
        for fut in as_completed(futures):
            name, code, secs = fut.result()
            done += 1
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"[{done}/{len(files)}] {name}: {status} in {secs:.0f} s", flush=True)


if __name__ == "__main__":
    main()
