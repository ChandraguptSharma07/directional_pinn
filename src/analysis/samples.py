"""Assembling the samples of the central experiment (Phases 5-8).

One row per (PINN run, horizon, sampled time), with the targets, the residual
predictors along the candidate directions, and the control alignments (random
directions, time-shuffled directions). Used by experiments/07 and 08.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.future_error import created_error, future_error, horizon_steps
from src.analysis.projections import decompose_residual
from src.dynamics.tangent_dynamics import propagators_over

STRIDE = 5  # statistics use every 5th grid point (spacing 0.01)
MIN_POINTS_PER_RUN = 10  # a horizon needs at least this many usable times per run
N_RANDOM, N_SHUFFLE = 20, 20
DIRECTIONS = ("b1", "s1")


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted((runs_dir / "runs").iterdir()):
        summary = json.loads((run_dir / "metrics.json").read_text())
        trajectory = np.load(run_dir / "trajectory.npz")
        residuals = np.load(run_dir / "residuals.npz")
        runs.append({"name": run_dir.name, "summary": summary, "t": trajectory["t"], "error": trajectory["error"],
                     "pinn": trajectory["pinn"], "reference": trajectory["reference"],
                     "residual": residuals["residual"]})
    return runs


def unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def join_free_indices(t: np.ndarray, indices: np.ndarray, segment_length: float | None) -> np.ndarray:
    """Drop exact internal joins, where a piecewise derivative is two-sided."""
    if segment_length is None:
        return indices
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    ratio = t[indices] / segment_length
    is_join = (t[indices] > 0) & np.isclose(ratio, np.round(ratio), atol=1e-8, rtol=0)
    return indices[~is_join]


def build_samples(runs: list[dict], lyapunov: dict, horizons: list[float],
                  exclude_joins: float | None = None) -> pd.DataFrame:
    """One row per (run, horizon, sampled time) with targets, predictors and control alignments."""
    tables = []
    for run_number, run in enumerate(runs):
        if not run["summary"]["finite"]:
            continue
        lyap = lyapunov[run["summary"]["window"]]
        t, error, residual = run["t"], run["error"], run["residual"]
        n, h = len(t), t[1] - t[0]
        if not np.allclose(lyap["t"][:n], t, atol=1e-12):
            raise ValueError(f"{run['name']}: time grids of run and Lyapunov data differ")
        step_props = lyap["step_propagators"][: n - 1]
        for k, dt in enumerate(horizons):
            m = horizon_steps(dt, h)
            j = np.arange(0, n - m, STRIDE)
            j = join_free_indices(t, j, exclude_joins)
            if len(j) < MIN_POINTS_PER_RUN:  # horizon too long for the PINN window
                continue
            phi = propagators_over(step_props, m)[j]
            r = residual[j]
            V = lyap[f"V_{dt:g}"][j]
            directions = {
                "b1": lyap["Q"][j, :, 0], "b3": lyap["Q"][j, :, 2],
                "s1": V[:, :, 0], "s3": V[:, :, 2], "flow": lyap["flow_direction"][j],
            }
            R = np.linalg.norm(r, axis=1)
            floor = 1e-6 * np.median(np.linalg.norm(residual, axis=1))
            row = {
                "run": run["name"], "window": run["summary"]["window"], "seed": run["summary"]["seed"],
                "horizon": dt, "t": t[j],
                "log_E_future": np.log(future_error(error, m)[j]),
                "log_N": np.log(created_error(error, propagators_over(step_props, m), m)[j]),
                "log_E_now": np.log(np.linalg.norm(error[j], axis=1)),
                "log_R": np.log(R),
                "log_amp": np.log(np.linalg.norm(np.einsum("jab,jb->ja", phi, r), axis=1) / R),
            }
            for name, u in directions.items():
                parts = decompose_residual(r, u)
                row[f"A_{name}"] = parts.alignment
                row[f"log_Ru_{name}"] = np.log(parts.R_u + floor)
                row[f"log_Rperp_{name}"] = np.log(parts.R_perp + floor)
            for d in range(N_RANDOM):
                rng = np.random.default_rng([d, run_number, k])
                row[f"A_random_{d}"] = decompose_residual(r, unit_rows(rng.normal(size=r.shape))).alignment
            for d in range(N_SHUFFLE):
                rng = np.random.default_rng([1000 + d, run_number, k])
                permutation = rng.permutation(len(j))
                for name in DIRECTIONS:
                    row[f"A_shuffled_{name}_{d}"] = decompose_residual(r, directions[name][permutation]).alignment
            tables.append(pd.DataFrame(row))
    return pd.concat(tables, ignore_index=True)
