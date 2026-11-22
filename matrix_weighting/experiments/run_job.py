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
    if job.get("segments"):
        return run_sequential(job, out, system, settings, pairs, x0, seeds)
    tg, te = grid(settings.t_final, settings.n_grid), grid(settings.t_final, settings.n_eval)
    refs = {ic["id"]: (reference(system, ic["x0"], tg), reference(system, ic["x0"], te)) for ic in job["ics"]}
    ref_grid = np.stack([refs[ic["id"]][0] for ic, _ in pairs])
    ref_eval = np.stack([refs[ic["id"]][1] for ic, _ in pairs])

    wall = time.perf_counter()
    res = train_batch(system, job["method"], x0, seeds, job["out_mean"], job["out_scale"], settings,
                      ref_grid=ref_grid, ref_eval=ref_eval, checkpoint=out / "model.pt")
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
        "resumed_from_checkpoint": res["resumed_from_checkpoint"],
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


def run_sequential(job: dict, out: Path, system, settings: Settings, pairs, x0: np.ndarray, seeds: list[int]) -> dict:
    """Time-marching (plan amendment 2): ``segments`` consecutive windows of length settings.t_final.

    Each segment trains fresh networks whose initial state is the previous segment's predicted end state, so
    errors carry over exactly as in a real initial-value solve. The reference is used only for the final
    metrics. Oracle weights are not available here.
    """
    if job["method"] in ("S-orc", "M-orc", "M-final-orc", "D-orc"):
        raise ValueError("sequential jobs support only reference-free weights")
    S, L = int(job["segments"]), float(settings.t_final)
    te = grid(L, settings.n_eval)
    t_all = np.concatenate([te] + [te[1:] + s * L for s in range(1, S)])
    ref_all = np.stack([reference(system, ic["x0"], t_all) for ic, _ in pairs])
    x_cur, xs, rs, hists, runtime = x0.copy(), [], [], [], 0.0
    wall = time.perf_counter()
    for s in range(S):
        res = train_batch(system, job["method"], x_cur, [sd + 100_000 * s for sd in seeds], job["out_mean"],
                          job["out_scale"], settings)
        xs.append(res["x_eval"] if s == 0 else res["x_eval"][:, 1:])
        rs.append(res["r_eval"] if s == 0 else res["r_eval"][:, 1:])
        x_cur = res["x_eval"][:, -1].copy()
        runtime += res["runtime_seconds"]
        hists.append(res["history"])
        print(f"segment {s + 1}/{S} done", flush=True)
    x_all, r_all = np.concatenate(xs, axis=1), np.concatenate(rs, axis=1)
    summary = {"job": job, "pairs": [{"ic": ic["id"], "seed": seed} for ic, seed in pairs],
               "metrics": trajectory_metrics(x_all, ref_all, r_all), "history": hists, "runtime_seconds": runtime,
               "wall_seconds": time.perf_counter() - wall, "device": settings.device,
               "torch_version": torch.__version__, "numpy_version": np.__version__, "host": platform.node(),
               "python": platform.python_version()}
    np.savez_compressed(out / "data.npz", x_eval=x_all, r_eval=r_all, ref_eval=ref_all, t_eval=t_all)
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
