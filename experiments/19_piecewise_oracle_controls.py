"""Check whether oracle amplification adds information beyond elapsed time.

This is a post-hoc diagnostic, separate from the fixed original decision rule.
The oracle uses the tangent propagator along the known reference trajectory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.metrics import bootstrap_delta_r_squared, leave_one_group_out, r_squared
from src.analysis.samples import build_samples, load_runs
from src.utils.reproducibility import load_config, save_json


def assess(rows, seed: int) -> dict:
    y = rows["log_E_future"].to_numpy()
    groups = rows["window"].to_numpy()
    clusters = rows["run"].to_numpy()
    log_r = rows["log_R"].to_numpy()
    t = rows["t"].to_numpy()
    amp = rows["log_amp"].to_numpy()
    alignment = rows["A_s1"].to_numpy()
    now = rows["log_E_now"].to_numpy()
    bases = {
        "residual_time": np.column_stack([log_r, t, t**2]),
        "residual_time_current_error": np.column_stack([log_r, t, t**2, now]),
    }
    output = {"n_samples": len(rows), "n_runs": rows["run"].nunique()}
    for name, X in bases.items():
        p_base = leave_one_group_out(X, y, groups)
        candidates = {"oracle_amplification": amp, "s1_alignment": alignment}
        entry = {"r2_base": r_squared(y, p_base)}
        for candidate, values in candidates.items():
            p_new = leave_one_group_out(np.column_stack([X, values]), y, groups)
            entry[candidate] = bootstrap_delta_r_squared(y, p_base, p_new, clusters,
                                                          n_boot=2000, seed=seed)
        output[name] = entry
    return output


def main() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    horizons = [float(h) for h in config["lorenz_lyapunov"]["horizons"]]
    runs = load_runs(ROOT / "results" / "piecewise_40runs_analysis_runs")
    lyapunov = {w: dict(np.load(ROOT / "results" / "interpolation_rebuild" / "lyapunov"
                                  / "data" / f"lyapunov_window_{w}.npz")) for w in range(1, 9)}
    samples = build_samples(runs, lyapunov, horizons, exclude_joins=0.25)
    result = {"experiment": "19_piecewise_oracle_controls", "post_hoc": True,
              "join_times_excluded": True, "horizons": {}}
    for k, horizon in enumerate(horizons):
        rows = samples[samples["horizon"] == horizon]
        result["horizons"][f"{horizon:g}"] = {
            "all_windows": assess(rows, k),
            "without_window_1": assess(rows[rows["window"] != 1], 100 + k),
        }
    out = ROOT / "results" / "piecewise_40runs_oracle_controls"
    out.mkdir(parents=True, exist_ok=True)
    save_json(result, out / "metrics.json")
    for horizon, entry in result["horizons"].items():
        for cohort, values in entry.items():
            oracle = values["residual_time"]["oracle_amplification"]
            print(f"h={horizon}, {cohort}: base R2 {oracle['r2_base']:+.3f}, "
                  f"oracle delta {oracle['delta']:+.3f} [{oracle['ci_low']:+.3f}, {oracle['ci_high']:+.3f}]")
    print(json.dumps({"output": str(out / "metrics.json")}))


if __name__ == "__main__":
    main()
