"""One PINN run: train on a window, evaluate against the reference, save everything.

A job is a plain dict, so that jobs can be sent to worker processes:

    {"run_dir": ..., "window": k, "reference_file": ".../window_k.npz",
     "params": {"sigma": ..., "rho": ..., "beta": ...}, "train": {TrainConfig fields}}

Each run directory receives:

    config.yaml     the job (training settings, window, parameters)
    metadata.json   environment, git commit, run time
    model.pt        trained weights (state_dict)
    history.json    loss history
    trajectory.npz  t, reference state, PINN state, error e = x_hat - x_ref
    residuals.npz   t, residual r(t), time derivative of x_hat
    metrics.json    summary numbers
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.pinn.train import PINN, TrainConfig, evaluate, train_pinn, train_pinns
from src.systems.lorenz63 import LorenzParams
from src.utils.reproducibility import save_json, write_run_record

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_job(job: dict) -> dict:
    """Train one PINN, evaluate it on the reference grid inside [0, T], save, and return the summary."""
    cfg = TrainConfig.from_dict(job["train"])
    model, history = train_pinn(cfg, LorenzParams.from_dict(job["params"]))
    return finish_run(job, model, history)


def run_jobs_together(jobs: list[dict]) -> list[dict]:
    """Train several jobs as one stacked model (see train_pinns).

    The jobs may differ only in their seed and initial condition (window).
    """
    settings = [{**job["train"], "seed": None, "x0": None} for job in jobs]
    if any(s != settings[0] for s in settings) or len({str(job["params"]) for job in jobs}) != 1:
        raise ValueError("jobs trained together may differ only in their seed and initial condition")
    cfg = TrainConfig.from_dict(jobs[0]["train"])
    seeds = [int(job["train"]["seed"]) for job in jobs]
    x0s = [job["train"]["x0"] for job in jobs]
    models, histories = train_pinns(cfg, seeds, LorenzParams.from_dict(jobs[0]["params"]), x0s=x0s)
    return [finish_run(job, model, history) for job, model, history in zip(jobs, models, histories)]


def finish_run(job: dict, model: PINN, history: dict) -> dict:
    """Evaluate a trained PINN against the reference, save the run directory, return the summary."""
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig.from_dict(job["train"])
    params = LorenzParams.from_dict(job["params"])

    reference = np.load(job["reference_file"])
    inside = reference["t"] <= cfg.t_final + 1e-9
    t, x_ref = reference["t"][inside], reference["states"][inside]
    evaluation = evaluate(model, t, params)
    error = evaluation["state"] - x_ref
    size = np.linalg.norm(error, axis=1)
    residual_size = np.linalg.norm(evaluation["residual"], axis=1)

    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)
    np.savez(run_dir / "trajectory.npz", t=t, reference=x_ref, pinn=evaluation["state"], error=error)
    np.savez(run_dir / "residuals.npz", t=t, residual=evaluation["residual"], derivative=evaluation["derivative"])
    summary = {
        "window": job["window"],
        "seed": cfg.seed,
        "t_final": cfg.t_final,
        "lambda_ic": cfg.lambda_ic,
        "dtype": cfg.dtype,
        "final_loss": history["loss"][-1],
        "final_loss_ode": history["loss_ode"][-1],
        "final_loss_ic": history["loss_ic"][-1],
        "finite": bool(np.isfinite(history["loss"][-1]) and np.all(np.isfinite(error))),
        "initial_error": float(size[0]),
        "median_error": float(np.median(size)),
        "max_error": float(size.max()),
        "final_error": float(size[-1]),
        "fraction_error_below_1": float(np.mean(size < 1.0)),
        "median_residual": float(np.median(residual_size)),
        "runtime_seconds": history["runtime_seconds"],
    }
    save_json(summary, run_dir / "metrics.json")
    write_run_record(run_dir, job, REPO_ROOT, history["runtime_seconds"])
    return summary
