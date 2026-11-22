"""Dial study, step 1: reference solutions, propagators and finite-time directions.

For every setting of docs/dial_study_plan.md and every window 0-8:

* the reference trajectory on t = 0, 0.002, ..., 2. It is exact for family L.
  For family D it comes from DOP853 (rtol 1e-11, atol 1e-13) and is checked
  against a run with ten times looser tolerances;
* the step propagators Phi(t_{i+1}, t_i). They are exact (expm) for family L
  and come from the tangent integrator for family D;
* for each horizon dt: the singular values and right singular vectors of
  Phi(t+dt, t), and the integrated propagator K(t; dt).

No PINN is involved. The metrics also report the dial descriptives of plan
section 10 (stretching ratio sigma_1/sigma_2 and the ratio of K's singular
values over windows 1-8). These depend on the dynamics only.

    python experiments/24_dial_reference.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dynamics.lyapunov import singular_directions  # noqa: E402
from src.dynamics.tangent_dynamics import integrated_propagators, propagators_over, step_propagators  # noqa: E402
from src.solvers.reference_solver import integrate  # noqa: E402
from src.systems.dial_systems import DavisSkodje, StiffSaddle, dial_settings  # noqa: E402
from src.utils.reproducibility import save_json, write_run_record  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
T_FINAL, H = 2.0, 0.002
HORIZONS = (0.05, 0.1, 0.25, 0.5)
WINDOWS = range(9)


def reference_window(system, window: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reference states and step propagators for one window, plus check results."""
    x0 = system.initial_state(window)
    checks = {}
    if isinstance(system, StiffSaddle):
        states = system.exact_solution(t, window)
        steps = np.broadcast_to(expm(system.matrix * H), (len(t) - 1, 2, 2)).copy()
        numerical = integrate(system.rhs, x0, t, rtol=1e-12, atol=1e-14).y
        checks["exact_vs_dop853_max_rel"] = float(np.max(np.abs(numerical - states)) / np.max(np.abs(states)))
        # The exact formula is the reference; DOP853 only cross-checks it. On the stiffest
        # setting the explicit solver itself is accurate to about 1e-8, hence the threshold.
        checks["exact_vs_dop853_ok"] = checks["exact_vs_dop853_max_rel"] < 1e-7
        return states, steps, checks

    sol = integrate(system.rhs, x0, t, rtol=1e-11, atol=1e-13, dense_output=True)
    loose = integrate(system.rhs, x0, t, rtol=1e-10, atol=1e-12)
    checks["tolerance_max_abs_diff"] = float(np.max(np.abs(sol.y - loose.y)))
    checks["tolerance_ok"] = checks["tolerance_max_abs_diff"] < 1e-8
    checks["dense_vs_grid_max_abs_diff"] = float(np.max(np.abs(sol.at(t) - sol.y)))
    steps = step_propagators(lambda times: system.jacobian(sol.at(times)), t)
    # Finite-difference check of the propagator over dt = 0.1 from three start times.
    m = int(round(0.1 / H))
    phi = propagators_over(steps, m)
    worst = 0.0
    for j in (0, 300, 800):
        base = solve_ivp(system.rhs, (t[j], t[j + m]), sol.y[j], method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
        for k in range(2):
            eps = 1e-6 * np.eye(2)[k]
            moved = solve_ivp(system.rhs, (t[j], t[j + m]), sol.y[j] + eps, method="DOP853",
                              rtol=1e-12, atol=1e-14).y[:, -1]
            column = (moved - base) / 1e-6
            worst = max(worst, float(np.linalg.norm(column - phi[j][:, k]) / np.linalg.norm(phi[j][:, k])))
    checks["propagator_vs_finite_difference_max_rel"] = worst
    checks["propagator_ok"] = worst < 1e-4
    return sol.y, steps, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/dial_study/reference")
    parser.add_argument("--settings", nargs="*", default=None, help="setting names (default: all eight)")
    parser.add_argument("--windows", nargs="*", type=int, default=list(WINDOWS))
    args = parser.parse_args()
    start = time.perf_counter()
    t = np.round(np.arange(0.0, T_FINAL + H / 2, H), 10)
    summary = {"t_final": T_FINAL, "h": H, "horizons": list(HORIZONS), "settings": {}}
    all_ok = True
    for system in [s for s in dial_settings() if args.settings is None or s.name in args.settings]:
        setting_dir = args.out / system.name
        (setting_dir / "data").mkdir(parents=True, exist_ok=True)
        ratios = {dt: [] for dt in HORIZONS}
        k_ratios = {dt: [] for dt in HORIZONS}
        window_checks, scale_states = {}, []
        for window in args.windows:
            states, steps, checks = reference_window(system, window, t)
            window_checks[str(window)] = checks
            all_ok &= all(v for k, v in checks.items() if k.endswith("_ok"))
            scale_states.append(states)
            arrays = {"t": t, "states": states, "step_propagators": steps,
                      "flow_direction": system.rhs_batch(states) / np.linalg.norm(system.rhs_batch(states), axis=1,
                                                                                  keepdims=True)}
            for dt in HORIZONS:
                m = int(round(dt / H))
                phi = propagators_over(steps, m)
                sigma, V = singular_directions(phi)
                K = integrated_propagators(steps, m, H)
                k_sigma = np.linalg.svd(K, compute_uv=False)
                arrays[f"sigma_{dt:g}"], arrays[f"V_{dt:g}"], arrays[f"K_{dt:g}"] = sigma, V, K
                if window >= 1:
                    ratios[dt].append(sigma[:, 0] / sigma[:, 1])
                    k_ratios[dt].append(k_sigma[:, 0] / k_sigma[:, 1])
            np.savez(setting_dir / "data" / f"window_{window}.npz", **arrays)
        stacked = np.concatenate(scale_states)
        out_scale = np.maximum(stacked.std(axis=0), 0.1)
        descriptives = {}
        for dt in HORIZONS:
            r, kr = np.concatenate(ratios[dt]), np.concatenate(k_ratios[dt])
            descriptives[f"{dt:g}"] = {
                "sigma_ratio_p10_p50_p90": np.quantile(r, [0.1, 0.5, 0.9]).tolist(),
                "integrated_lever_p10_p50_p90": np.quantile(kr, [0.1, 0.5, 0.9]).tolist(),
            }
        summary["settings"][system.name] = {
            "system": system.to_dict(), "dial": system.dial,
            "out_mean": stacked.mean(axis=0).tolist(), "out_scale": out_scale.tolist(),
            "max_state_norm_by_window": {str(w): float(np.linalg.norm(s, axis=1).max())
                                         for w, s in zip(args.windows, scale_states)},
            "checks": window_checks, "descriptives": descriptives,
        }
        d01 = descriptives["0.1"]
        print(f"{system.name:22s} dial={system.dial:4g}  sigma1/sigma2 at dt=0.1 median "
              f"{d01['sigma_ratio_p10_p50_p90'][1]:9.2f}  integrated lever median "
              f"{d01['integrated_lever_p10_p50_p90'][1]:6.2f}", flush=True)
    summary["all_checks_passed"] = bool(all_ok)
    save_json(summary, args.out / "metrics.json")
    write_run_record(args.out, {"script": "experiments/24_dial_reference.py", "out": str(args.out)},
                     REPO_ROOT, time.perf_counter() - start)
    print("all checks passed:", all_ok)


if __name__ == "__main__":
    main()
