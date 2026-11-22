"""Window-level uncertainty for a frozen piecewise-PINN validation set.

Leave-one-window-out predictions follow the existing analysis. Bootstrap
clusters are the eight initial-condition windows, keeping their five seeds
together. This script is prepared before inspecting validation outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.metrics import bootstrap_delta_r_squared, leave_one_group_out
from src.analysis.samples import build_samples, load_runs
from src.utils.reproducibility import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--lyapunov", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    horizons = [float(x) for x in config["lorenz_lyapunov"]["horizons"]]
    runs = load_runs(args.runs)
    windows = sorted({run["summary"]["window"] for run in runs})
    lyapunov = {w: dict(np.load(args.lyapunov / "data" / f"lyapunov_window_{w}.npz")) for w in windows}
    samples = build_samples(runs, lyapunov, horizons, exclude_joins=0.25)
    outcome = {"experiment": "20_independent_window_analysis", "n_runs": len(runs),
               "n_windows": len(windows), "n_samples": len(samples),
               "bootstrap_unit": "initial-condition window (all seeds together)", "horizons": {}}
    for k, horizon in enumerate(horizons):
        rows = samples[samples["horizon"] == horizon]
        y = rows["log_E_future"].to_numpy()
        groups = rows["window"].to_numpy()
        log_r = rows["log_R"].to_numpy()
        t = rows["t"].to_numpy()
        bases = {"residual_only": log_r[:, None],
                 "residual_plus_time": np.column_stack([log_r, t, t**2])}
        features = {"b1_alignment": rows["A_b1"].to_numpy(),
                    "s1_alignment": rows["A_s1"].to_numpy(),
                    "oracle_amplification": rows["log_amp"].to_numpy()}
        entry = {"n_samples": len(rows), "models": {}}
        for base_name, X in bases.items():
            p0 = leave_one_group_out(X, y, groups)
            entry["models"][base_name] = {}
            for feature_name, feature in features.items():
                p1 = leave_one_group_out(np.column_stack([X, feature]), y, groups)
                entry["models"][base_name][feature_name] = bootstrap_delta_r_squared(
                    y, p0, p1, groups, n_boot=5000, seed=100 * k + len(base_name) + len(feature_name))
        outcome["horizons"][f"{horizon:g}"] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(outcome, args.out / "metrics.json")
    for h, entry in outcome["horizons"].items():
        base = entry["models"]["residual_only"]
        timed = entry["models"]["residual_plus_time"]
        print(f"h={h}: b1 {base['b1_alignment']['delta']:+.3f}, "
              f"s1 {base['s1_alignment']['delta']:+.3f}, "
              f"oracle+time {timed['oracle_amplification']['delta']:+.3f}")
    print(json.dumps({"output": str(args.out / "metrics.json")}))


if __name__ == "__main__":
    main()
