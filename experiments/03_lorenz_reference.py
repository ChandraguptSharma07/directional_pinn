"""Phase 2: high-accuracy Lorenz-63 reference trajectories, and how long they can be trusted.

Run from the repository root:

    python experiments/03_lorenz_reference.py

For every window (initial condition) the reference is re-solved with a ladder
of tolerances and with a different method (LSODA) and compared with the
strictest setting. Because Lorenz-63 is chaotic, any two numerical solutions
eventually separate; the script records, for each window, how long the
reference stays within `acceptance` of the strictest solution, and fails
(exit status 1) if that is shorter than `required_horizon`.

Outputs in results/lorenz_reference/: metrics.json, figures, and one
data/window_<k>.npz per window (reference states, past, deviations), which
later phases read.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/03_lorenz_reference.py` import src

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.plots import COLORS, apply_style, ordinal_colors, save_figure
from src.solvers.lorenz_reference import build_windows_from_config, solver_settings
from src.solvers.reference_solver import setting_label, tolerance_study, trusted_until
from src.systems.lorenz63 import LorenzParams, lorenz_rhs, verify_lorenz_jacobian
from src.utils.checks import Checks
from src.utils.reproducibility import load_config, save_json, write_run_record

LYAPUNOV_1 = 0.906  # commonly quoted largest Lyapunov exponent, used only for a guide line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: Lorenz-63 reference trajectories and tolerance study.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "lorenz_reference")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_config(args.config)
    params = LorenzParams.from_dict(config["lorenz"]["params"])
    cfg = config["lorenz_reference"]
    if not verify_lorenz_jacobian(params).passed:  # the gate from Phase 1
        print("Jacobian check failed; not proceeding.")
        return 1
    data_dir, figures_dir = args.out / "data", args.out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    checks = Checks()

    solver = solver_settings(cfg["solver"])
    strictest = solver_settings(cfg["strictest"])
    ladder = [solver_settings(setting) for setting in cfg["ladder"]]
    acceptance, required = float(cfg["acceptance"]), float(cfg["required_horizon"])
    cross_acceptance = float(cfg["cross_check_acceptance"])
    reference_label = setting_label(solver)
    other_method = sorted((s for s in ladder if s["method"] != solver["method"]), key=lambda s: -s["rtol"])
    cross_label = setting_label(other_method[-1])  # the other method at its tightest tolerance
    if reference_label not in {setting_label(s) for s in ladder}:
        raise ValueError("the reference setting must be part of the ladder")
    rhs = lambda t, state: lorenz_rhs(t, state, params)

    windows = build_windows_from_config(params, cfg)
    t = windows[0].reference.t
    past_t = np.linspace(-float(cfg["spinup"]), 0.0, int(round(float(cfg["spinup"]) / float(cfg["dt"]))) + 1)
    per_window, all_deviations = [], []
    for window in windows:
        _, deviations = tolerance_study(rhs, window.x0, t, ladder, strictest)
        all_deviations.append(deviations)
        trust = trusted_until(t, deviations[reference_label], acceptance)
        cross_trust = trusted_until(t, deviations[cross_label], cross_acceptance)
        checks.add(f"window {window.index}: reference within {acceptance:g} of strictest up to t={required:g}"
                   " (max deviation)", deviations[reference_label][t <= required].max(), acceptance)
        checks.add(f"window {window.index}: {cross_label} within {cross_acceptance:g} of strictest up to"
                   f" t={required:g} (max deviation)", deviations[cross_label][t <= required].max(), cross_acceptance)
        for method_settings in ([s for s in ladder if s["method"] == solver["method"]], other_method):
            at_required = [deviations[setting_label(s)][np.searchsorted(t, required)] for s in method_settings]
            checks.add(f"window {window.index}: {method_settings[0]['method']} deviation at t={required:g} shrinks"
                       " as rtol tightens (violations)", float(np.sum(np.diff(at_required) >= 0)), 0.0)
        np.savez(
            data_dir / f"window_{window.index}.npz",
            t=t,
            states=window.reference.y,
            x0=window.x0,
            start_in_long_run=window.start_in_long_run,
            past_t=past_t,
            past_states=window.past(past_t),
            **{f"deviation {label}": value for label, value in deviations.items()},
        )
        per_window.append(
            {
                "index": window.index,
                "role": "pilot" if window.index == 0 else "analysis",
                "start_in_long_run": window.start_in_long_run,
                "x0": window.x0,
                "trusted_until": trust,
                "trusted_until_cross_check": cross_trust,
                "deviation_at_required_horizon": {
                    label: dev[np.searchsorted(t, required)] for label, dev in deviations.items()
                },
                "reference_nfev": window.reference.stats["nfev"],
            }
        )

    plot_attractor(windows, cfg, required, figures_dir)
    plot_tolerance_study(t, all_deviations, ladder, reference_label, cross_label, acceptance, required, per_window,
                         figures_dir)

    metrics = {
        "experiment": "03_lorenz_reference",
        "solver": solver,
        "strictest": strictest,
        "ladder": ladder,
        "acceptance": acceptance,
        "required_horizon": required,
        "min_trusted_until": min(w["trusted_until"] for w in per_window),
        "windows": per_window,
        "checks": checks.records,
        "all_checks_passed": checks.all_passed,
    }
    save_json(metrics, args.out / "metrics.json")
    write_run_record(args.out, {key: config[key] for key in ("lorenz", "lorenz_reference")}, REPO_ROOT,
                     time.perf_counter() - start)

    print(f"Results written to {args.out}")
    print(checks.report())
    for w in per_window:
        print(f"window {w['index']} ({w['role']}): x0 = {np.round(w['x0'], 3)}, reference trusted until "
              f"t = {w['trusted_until']:.2f} (cross-check {w['trusted_until_cross_check']:.2f})")
    return 0 if checks.all_passed else 1


def plot_attractor(windows, cfg, required, figures_dir) -> None:
    long_run = windows[0].long_run
    t_long = np.arange(float(cfg["transient"]), windows[-1].start_in_long_run, 0.005)
    states = long_run.at(t_long)
    fig = plt.figure(figsize=(12.0, 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*states.T, color=COLORS["axis"], lw=0.4)
    shown = windows[1]
    segment = shown.reference.y[shown.reference.t <= required]
    ax.plot(*segment.T, color=COLORS["ink"], lw=1.2, label=f"window {shown.index}, $0 \\leq t \\leq {required:g}$")
    for window in windows:
        ax.scatter(*window.x0, color=COLORS["unstable"], s=18, depthshade=False)
        ax.text(*window.x0, f" {window.index}", fontsize=8, color=COLORS["ink_secondary"])
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_title("(a) Attractor (grey) and initial conditions of the windows (orange)")
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=-60)

    ax = fig.add_subplot(1, 2, 2)
    mask = shown.reference.t <= required
    for k, (name, color) in enumerate((("x", COLORS["unstable"]), ("y", COLORS["stable"]), ("z", COLORS["third"]))):
        ax.plot(shown.reference.t[mask], shown.reference.y[mask, k], color=color, label=name)
    ax.set(xlabel="t", ylabel="state")
    ax.set_title(f"(b) Reference trajectory of window {shown.index}")
    ax.legend(loc="upper right", ncol=3)
    save_figure(fig, figures_dir, "fig01_attractor_and_windows")


def plot_tolerance_study(t, all_deviations, ladder, reference_label, cross_label, acceptance, required, per_window,
                         figures_dir) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    floor = 1e-17  # deviations of exactly 0 cannot be drawn on a log scale
    ax = axes[0]
    shown = 1
    other_method = cross_label.split()[0]
    dop = [s for s in ladder if s["method"] != other_method]
    for setting, color in zip(dop, ordinal_colors(len(dop))):
        label = setting_label(setting)
        ax.plot(t, np.maximum(all_deviations[shown][label], floor), color=color, lw=1.2,
                label=label + (" (reference)" if label == reference_label else ""))
    for setting in (s for s in ladder if s["method"] == other_method):
        label = setting_label(setting)
        tightest = label == cross_label
        ax.plot(t, np.maximum(all_deviations[shown][label], floor), color=COLORS["third"], lw=1.2,
                alpha=1.0 if tightest else 0.45, label=label + " (other method)")
    guide = 1e-13 * np.exp(LYAPUNOV_1 * t)
    ax.plot(t, guide, color=COLORS["muted"], lw=1.0, label="$10^{-13} e^{0.906\\,t}$ (growth at the Lyapunov rate)")
    ax.axhline(acceptance, color=COLORS["unstable"], lw=1.0, label=f"acceptance {acceptance:g}")
    ax.axvline(required, color=COLORS["ink_secondary"], lw=1.0)
    ax.set_yscale("log")
    ax.set(ylim=(1e-16, 1e2), xlabel="t", ylabel="$\\Vert x_{setting}(t) - x_{strictest}(t)\\Vert$")
    ax.set_title(f"(a) Window {shown}: distance from the strictest solution (DOP853 rtol=1e-13)")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    for k, deviations in enumerate(all_deviations):
        ax.plot(t, np.maximum(deviations[reference_label], floor), color=COLORS["ink"] if k else COLORS["muted"],
                lw=0.9, alpha=0.8)
    ax.axhline(acceptance, color=COLORS["unstable"], lw=1.0, label=f"acceptance {acceptance:g}")
    ax.axvline(required, color=COLORS["ink_secondary"], lw=1.0, label=f"required horizon t = {required:g}")
    ax.set_yscale("log")
    ax.set(ylim=(1e-16, 1e2), xlabel="t", ylabel="$\\Vert x_{reference}(t) - x_{strictest}(t)\\Vert$")
    trusted = [w["trusted_until"] for w in per_window]
    ax.set_title(f"(b) All {len(per_window)} windows: reference trusted until t = {min(trusted):.1f} to {max(trusted):.1f}")
    ax.legend(loc="upper left")
    save_figure(fig, figures_dir, "fig02_tolerance_study")


if __name__ == "__main__":
    sys.exit(main())
