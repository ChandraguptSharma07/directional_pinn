"""Exploratory follow-up: does interpolating and averaging residuals help?

Sample each trained PINN residual every 0.05 time units, reconstruct it linearly
or by sample-and-hold, and take trailing averages over 0.1 and 0.25. Predictions
are evaluated at sampled knots so that linear interpolation never uses residual
data from after the prediction time.
Evaluate interpolation fidelity against the original fine-grid residual and
leave-one-initial-condition-window-out prediction of future trajectory error.

Requires the untracked trajectory.npz and residuals.npz files produced by
experiments/06_pinn_runs.py. Run from the repository root:

    python experiments/09_residual_interpolation.py
    python experiments/09_residual_interpolation.py --runs results/pinn_runs_lambda10 \
        --out results/interpolation_lambda10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.analysis.interpolation import reconstruct_residual, sampled_knot_mean_magnitude, trailing_mean_magnitude

SAMPLE_SPACING = 0.05
WIDTHS = (0.1, 0.25)
HORIZONS = (0.05, 0.1, 0.25, 0.5)
N_BOOT = 2000
EPS = 1e-12


def load_runs(directory: Path) -> list[dict]:
    run_dirs = sorted((directory / "runs").glob("w*_s*"))
    if not run_dirs:
        raise FileNotFoundError(f"no PINN runs under {directory / 'runs'}")
    missing = [str(d / name) for d in run_dirs for name in ("trajectory.npz", "residuals.npz")
               if not (d / name).exists()]
    if missing:
        raise FileNotFoundError(f"raw arrays are absent ({len(missing)} files); rerun experiments/06_pinn_runs.py to regenerate them. First missing: {missing[0]}")
    runs = []
    for d in run_dirs:
        summary = json.loads((d / "metrics.json").read_text())
        if not summary["finite"]:
            continue
        with np.load(d / "trajectory.npz") as trajectory, np.load(d / "residuals.npz") as residuals:
            runs.append({"name": d.name, "window": summary["window"],
                         "t": trajectory["t"].copy(), "error": trajectory["error"].copy(),
                         "residual": residuals["residual"].copy()})
    if not runs:
        raise ValueError("no finite runs")
    return runs


def build_data(runs: list[dict]) -> tuple[dict[float, dict], dict]:
    tables = {h: {"y": [], "group": [], "run": [], "point": [], "time": [],
                  **{f"linear_{w:g}": [] for w in WIDTHS},
                  **{f"hold_{w:g}": [] for w in WIDTHS},
                  **{f"box_{w:g}": [] for w in WIDTHS},
                  **{f"dense_{w:g}": [] for w in WIDTHS}} for h in HORIZONS}
    fidelity = {f"{w:g}": {"linear": [], "hold": []} for w in WIDTHS}
    for run in runs:
        t, residual, error = run["t"], run["residual"], run["error"]
        h = t[1] - t[0]
        stride = int(round(SAMPLE_SPACING / h))
        if stride < 1 or not np.isclose(stride * h, SAMPLE_SPACING, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{run['name']}: sample spacing incompatible with grid")
        recon = {method: reconstruct_residual(t, residual, stride, method) for method in ("linear", "hold")}
        averages = {method: {w: trailing_mean_magnitude(t, values, w) for w in WIDTHS}
                    for method, values in {**recon, "dense": residual}.items()}
        for w in WIDTHS:
            valid = np.isfinite(averages["dense"][w])
            reference = averages["dense"][w][valid]
            for method in recon:
                estimate = averages[method][w][valid]
                fidelity[f"{w:g}"][method].append(float(np.mean(np.abs(estimate - reference)) /
                                                      max(np.mean(reference), EPS)))
        for horizon in HORIZONS:
            steps = int(round(horizon / h))
            if not np.isclose(steps * h, horizon, rtol=1e-9, atol=1e-12):
                raise ValueError(f"{run['name']}: horizon incompatible with grid")
            start = int(round(max(WIDTHS) / h))
            if start % stride:
                start += stride - start % stride
            j = np.arange(start, len(t) - steps, stride)
            if not len(j):
                continue
            table = tables[horizon]
            n = len(j)
            table["y"].append(np.log(np.maximum(np.linalg.norm(error[j + steps], axis=1), EPS)))
            table["group"].append(np.full(n, run["window"]))
            table["run"].append(np.full(n, run["name"]))
            table["point"].append(np.log(np.maximum(np.linalg.norm(recon["linear"][j], axis=1), EPS)))
            table["time"].append(t[j] / (t[-1] - t[0]))
            for w in WIDTHS:
                for method in averages:
                    table[f"{method}_{w:g}"].append(np.log(np.maximum(averages[method][w][j], EPS)))
                width_steps = int(round(w / h))
                sampled_box = sampled_knot_mean_magnitude(residual, j, width_steps, stride)
                table[f"box_{w:g}"].append(np.log(np.maximum(sampled_box, EPS)))
    return ({h: {key: np.concatenate(chunks) for key, chunks in table.items()}
             for h, table in tables.items() if table["y"]}, fidelity)


def leave_one_window_out(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.empty_like(y)
    for group in np.unique(groups):
        test = groups == group
        coefficients = np.linalg.lstsq(np.column_stack([np.ones((~test).sum()), X[~test]]), y[~test], rcond=None)[0]
        prediction[test] = np.column_stack([np.ones(test.sum()), X[test]]) @ coefficients
    return prediction


def score(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {"r2": float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)),
            "mae_log_error": float(np.mean(np.abs(y - prediction)))}


def window_bootstrap_gain(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                          groups: np.ndarray, seed: int) -> dict:
    """Resample independent initial-condition windows, keeping all seeds and times together."""
    ids = np.unique(groups)
    parts = [(y[groups == g], baseline[groups == g], candidate[groups == g]) for g in ids]
    rng = np.random.default_rng(seed)
    per_window = {}
    for group, (yy, bb, cc) in zip(ids, parts):
        base_sse = np.sum((yy - bb) ** 2)
        candidate_sse = np.sum((yy - cc) ** 2)
        per_window[str(group)] = float((base_sse - candidate_sse) / base_sse)
    draws = []
    for _ in range(N_BOOT):
        chosen = rng.integers(len(ids), size=len(ids))
        yy, bb, cc = (np.concatenate([parts[k][i] for k in chosen]) for i in range(3))
        if np.var(yy) > 0:
            draws.append(score(yy, cc)["r2"] - score(yy, bb)["r2"])
    return {"delta_r2": score(y, candidate)["r2"] - score(y, baseline)["r2"],
            "ci95_window_bootstrap": np.quantile(draws, [0.025, 0.975]).tolist(),
            "positive_windows": sum(value > 0 for value in per_window.values()),
            "per_window_relative_sse_reduction": per_window}


def evaluate(tables: dict[float, dict], fidelity: dict, n_runs: int) -> dict:
    out = {"experiment": "09_residual_interpolation", "exploratory": True, "n_runs": n_runs,
           "sample_spacing": SAMPLE_SPACING, "prediction_times": "sampled residual knots only",
           "trailing_widths": WIDTHS,
           "fidelity_relative_mae_by_run": {w: {m: {"mean": float(np.mean(v)), "per_run": v}
                                                     for m, v in methods.items()}
                                            for w, methods in fidelity.items()}, "horizons": {}}
    for k, (horizon, table) in enumerate(tables.items()):
        y, groups = table["y"], table["group"]
        base_X = np.column_stack([table["point"], table["time"]])
        baseline = leave_one_window_out(base_X, y, groups)
        entry = {"n_samples": len(y), "n_windows": len(np.unique(groups)),
                 "baseline_log_pointwise_plus_time": score(y, baseline), "models": {},
                 "paired_comparisons": {}}
        predictions = {}
        for w in WIDTHS:
            for method in ("linear", "hold", "box", "dense"):
                name = f"{method}_{w:g}"
                candidate = leave_one_window_out(np.column_stack([base_X, table[name]]), y, groups)
                predictions[name] = candidate
                entry["models"][name] = {**score(y, candidate),
                                         **window_bootstrap_gain(y, baseline, candidate, groups, seed=k)}
            linear = predictions[f"linear_{w:g}"]
            for control in ("hold", "box", "dense"):
                entry["paired_comparisons"][f"linear_minus_{control}_{w:g}"] = window_bootstrap_gain(
                    y, predictions[f"{control}_{w:g}"], linear, groups, seed=k + 100
                )
        out["horizons"][f"{horizon:g}"] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "results" / "pinn_runs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "interpolation")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        runs = load_runs(args.runs)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    tables, fidelity = build_data(runs)
    result = evaluate(tables, fidelity, len(runs))
    args.out.mkdir(parents=True, exist_ok=True)
    from src.utils.reproducibility import save_json, write_run_record

    save_json(result, args.out / "metrics.json")
    write_run_record(args.out, {"runs": str(args.runs), "sample_spacing": SAMPLE_SPACING,
                                "widths": WIDTHS, "horizons": HORIZONS}, REPO_ROOT, time.perf_counter() - started)
    for horizon, entry in result["horizons"].items():
        print(f"horizon {horizon}: baseline R² {entry['baseline_log_pointwise_plus_time']['r2']:+.3f}")
        for name, model in entry["models"].items():
            lo, hi = model["ci95_window_bootstrap"]
            print(f"  {name}: ΔR² {model['delta_r2']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for name, comparison in entry["paired_comparisons"].items():
            lo, hi = comparison["ci95_window_bootstrap"]
            print(f"  {name}: ΔR² {comparison['delta_r2']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
