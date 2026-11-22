"""Run one job: one system, one window length, one weighting method, a batch of networks.

A job file (JSON) holds the system, the method, the settings, the initial
conditions (each with an id), the seeds and the output directory (relative to
the matrix_weighting/ folder). Every (initial condition, seed) pair is one
network. The script writes summary.json (metrics, history, provenance) and
data.npz (trajectories; ignored by git). A finished job is skipped.

    python experiments/run_job.py jobs/gate0/saddle_k64_T2_M-orc.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mw.model import init_seed  # noqa: E402
from mw.reference import reference  # noqa: E402
from mw.systems import Lorenz63, system_from_dict  # noqa: E402
from mw.train import Settings, grid, train_batch, trajectory_metrics  # noqa: E402


def run(job: dict, root: Path = ROOT, device: str | None = None) -> dict | None:
    out = root / job["out_dir"]
    if (out / "summary.json").exists():
        return None
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(job.get("threads", 1)))
    system = system_from_dict(job["system"])
    settings = Settings.from_dict({**job["settings"], **({"device": device} if device else {})})
    pairs = [(ic, seed) for ic in job["ics"] for seed in job["seeds"]]
    x0 = np.array([ic["x0"] for ic, _ in pairs], dtype=float)
    seeds = [init_seed(seed, ic["id"]) for ic, seed in pairs]
    tg, te = grid(settings.t_final, settings.n_grid), grid(settings.t_final, settings.n_eval)
    refs = {ic["id"]: (reference(system, ic["x0"], tg), reference(system, ic["x0"], te)) for ic in job["ics"]}
    ref_grid = np.stack([refs[ic["id"]][0] for ic, _ in pairs])
    ref_eval = np.stack([refs[ic["id"]][1] for ic, _ in pairs])

    wall = time.perf_counter()
    res = train_batch(system, job["method"], x0, seeds, job["out_mean"], job["out_scale"], settings,
                      ref_grid=ref_grid, ref_eval=ref_eval)
    metrics = res["metrics"]
    extra = {}
    if isinstance(system, Lorenz63) and system.k > 0:
        extra["metrics_lorenz_part"] = trajectory_metrics(res["x_eval"], ref_eval, None, components=slice(0, 3))
    summary = {
        "job": job,
        "pairs": [{"ic": ic["id"], "seed": seed} for ic, seed in pairs],
        "metrics": metrics,
        **extra,
        "history": res["history"],
        "runtime_seconds": res["runtime_seconds"],
        "wall_seconds": time.perf_counter() - wall,
        "device": settings.device,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "host": platform.node(),
        "python": platform.python_version(),
    }
    np.savez_compressed(out / "data.npz", x_eval=res["x_eval"], r_eval=res["r_eval"], ref_eval=ref_eval,
                        t_eval=te)
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary) + "\n")
    tmp.replace(out / "summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    job = json.loads(args.job.read_text())
    if args.threads:
        job["threads"] = args.threads
    s = run(job, device=args.device)
    if s is None:
        print(f"skip (done): {job['out_dir']}")
        return
    m = s["metrics"]
    e = np.log10(np.clip(np.array(m["E_rms"], dtype=float), 1e-12, 1e3))
    print(f"done {job['out_dir']}: median log10 E_rms {np.median(e):.3f}, runtime {s['runtime_seconds']:.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
