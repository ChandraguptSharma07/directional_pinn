"""Phase 1: the Lorenz-63 equations and their analytic Jacobian, verified numerically.

Run from the repository root:

    python experiments/02_lorenz_jacobian.py
    python experiments/02_lorenz_jacobian.py --config config/default.yaml --out results/lorenz_jacobian

1. Jacobian check: the analytic Jacobian is compared with complex-step and
   central-difference Jacobians at the three fixed points and at random states
   around the attractor. A Jacobian with one deliberate sign error goes through
   the same check, to show that the check can fail.
2. Fixed points: the right-hand side vanishes there, and the eigenvalues of
   the Jacobian match their closed forms.
3. Stability of C+ and C- as rho varies: the largest real part of their
   eigenvalues crosses zero at the Hopf value rho_H.

All comparisons are saved in metrics.json under "checks"; the script exits
with status 1 if any fails. See docs/phase1_lorenz_jacobian.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/02_lorenz_jacobian.py` import src

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from src.analysis.plots import COLORS, apply_style, save_figure
from src.systems.lorenz63 import (
    LorenzParams,
    lorenz_divergence,
    lorenz_fixed_points,
    lorenz_jacobian,
    lorenz_rhs,
    verify_lorenz_jacobian,
)
from src.utils.checks import Checks
from src.utils.reproducibility import load_config, save_csv, save_json, write_run_record


def jacobian_with_sign_error(state: np.ndarray, params: LorenzParams) -> np.ndarray:
    """The analytic Jacobian with one sign error (+x instead of -x in row 2, column 3).

    A typical transcription mistake, used as a control: the check must reject it.
    """
    J = lorenz_jacobian(state, params)
    J[..., 1, 2] *= -1.0
    return J


def sort_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    """Sort by imaginary part, then real part (robust for conjugate pairs)."""
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    return eigenvalues[np.lexsort((eigenvalues.real, eigenvalues.imag))]


def closed_form_eigenvalues(params: LorenzParams) -> dict[str, np.ndarray]:
    """Eigenvalues of J at the fixed points from formulas that do not use lorenz_jacobian.

    Origin: J(0) splits into the block [[-sigma, sigma], [rho, -1]] and the entry -beta.
    C+ and C-: roots of lambda^3 + (sigma + beta + 1) lambda^2 + beta (sigma + rho) lambda
    + 2 sigma beta (rho - 1).
    """
    s, r, b = params.sigma, params.rho, params.beta
    root = np.sqrt((s + 1) ** 2 + 4 * s * (r - 1))
    closed = {"origin": np.array([(-(s + 1) + root) / 2, (-(s + 1) - root) / 2, -b])}
    if r > 1:
        cubic = np.roots([1.0, s + b + 1, b * (s + r), 2 * s * b * (r - 1)])
        closed["C+"] = closed["C-"] = cubic
    return closed


def largest_growth_rate_at_c_plus(params: LorenzParams) -> float:
    """Largest real part of the eigenvalues of J at C+ (C- has the same eigenvalues, by symmetry)."""
    point = lorenz_fixed_points(params)["C+"]
    return float(np.linalg.eigvals(lorenz_jacobian(point, params)).real.max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1: Lorenz-63 right-hand side and Jacobian checks.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "lorenz_jacobian")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_config(args.config)
    params = LorenzParams.from_dict(config["lorenz"]["params"])
    cfg = config["lorenz_jacobian_check"]
    data_dir, figures_dir = args.out / "data", args.out / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    checks = Checks()

    # 1. Jacobian check, and the same check applied to a deliberately wrong Jacobian.
    sampling = {"n_states": int(cfg["n_states"]), "seed": int(cfg["seed"]), "box": cfg["box"]}
    check = verify_lorenz_jacobian(params, **sampling)
    control = verify_lorenz_jacobian(params, jacobian=jacobian_with_sign_error, **sampling)
    for method, tolerance in check.tolerances.items():
        checks.add(f"Jacobian vs {method}, max relative error", check.errors[method].max(), tolerance)
    checks.add("control: Jacobian with a sign error is rejected (0 = rejected)", float(control.passed), 0.0)
    save_csv(
        data_dir / "jacobian_check.csv",
        {
            "x": check.states[:, 0],
            "y": check.states[:, 1],
            "z": check.states[:, 2],
            **{f"error_{method}": errors for method, errors in check.errors.items()},
            **{f"control_error_{method}": errors for method, errors in control.errors.items()},
        },
    )

    # 2. Fixed points and the eigenvalues of the Jacobian there.
    closed = closed_form_eigenvalues(params)
    fixed_points = {}
    for name, point in lorenz_fixed_points(params).items():
        eigenvalues = sort_eigenvalues(np.linalg.eigvals(lorenz_jacobian(point, params)))
        expected = sort_eigenvalues(closed[name])
        checks.add(f"{name}: max |f| at the fixed point", np.abs(lorenz_rhs(0.0, point, params)).max(), 1e-12)
        checks.add(f"{name}: eigenvalues vs closed form, max |difference|", np.abs(eigenvalues - expected).max(), 1e-10)
        fixed_points[name] = {
            "state": point,
            "eigenvalues": [{"real": ev.real, "imag": ev.imag} for ev in eigenvalues],
        }

    # 3. Stability of C+ and C- as rho varies.
    rho_grid = np.linspace(*map(float, cfg["rho_range"]), 400)
    growth = np.array([largest_growth_rate_at_c_plus(dataclasses.replace(params, rho=rho)) for rho in rho_grid])
    s, b = params.sigma, params.beta
    rho_hopf_closed = s * (s + b + 3) / (s - b - 1)
    rho_hopf = brentq(lambda rho: largest_growth_rate_at_c_plus(dataclasses.replace(params, rho=rho)),
                      rho_grid[0], rho_grid[-1], xtol=1e-13)
    checks.add("Hopf threshold: root of growth rate vs closed form", abs(rho_hopf - rho_hopf_closed), 1e-8)
    save_csv(data_dir / "c_plus_growth_vs_rho.csv", {"rho": rho_grid, "largest_real_part": growth})

    # Figure: eigenvalues at the fixed points, and stability of C+/- versus rho.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.axvline(0.0, color=COLORS["muted"], lw=1.0)
    ax.text(0.4, 1.5, "growth rate = 0", fontsize=8, color=COLORS["ink_secondary"], rotation=90, va="bottom")
    markers = {"origin": ("o", "origin"), "C+": ("s", "C+ and C- (identical)")}
    legend_handles = []
    for name, (marker, label) in markers.items():
        if name not in fixed_points:
            continue
        values = np.array([complex(ev["real"], ev["imag"]) for ev in fixed_points[name]["eigenvalues"]])
        filled = name == "origin"
        for ev in values:
            color = COLORS["unstable"] if ev.real > 0 else COLORS["stable"]
            ax.plot(ev.real, ev.imag, marker, color=color, ms=9, mfc=color if filled else "white", mew=2)
            text = f"{ev.real:.2f}" if abs(ev.imag) < 1e-12 else f"{ev.real:.3f} {'+' if ev.imag > 0 else '-'} {abs(ev.imag):.2f}i"
            ax.annotate(text, (ev.real, ev.imag), textcoords="offset points", xytext=(6, 8), fontsize=8,
                        color=COLORS["ink_secondary"])
        legend_handles.append(
            plt.Line2D([], [], ls="none", marker=marker, ms=9, mew=2, color=COLORS["ink_secondary"],
                       mfc=COLORS["ink_secondary"] if filled else "white", label=label)
        )
    ax.set(xlim=(-26.0, 16.0), ylim=(-13.0, 13.0), xlabel="real part (growth rate)",
           ylabel="imaginary part (rotation rate)")
    ax.set_title(f"(a) Eigenvalues of J at the fixed points ($\\rho$ = {params.rho:g})")
    ax.legend(handles=legend_handles, loc="upper left")
    ax.text(0.98, 0.03, "orange: growing direction\nblue: shrinking direction", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=COLORS["ink_secondary"])

    ax = axes[1]
    ax.axhline(0.0, color=COLORS["muted"], lw=1.0)
    ax.plot(rho_grid, growth, color=COLORS["ink"])
    ax.axvline(rho_hopf, color=COLORS["third"], lw=1.2)
    ax.text(rho_hopf, 0.05, f"$\\rho_H$ = {rho_hopf:.2f} ", transform=ax.get_xaxis_transform(), ha="right",
            fontsize=8.5, color=COLORS["ink_secondary"])
    ax.axvline(params.rho, color=COLORS["unstable"], lw=1.2)
    ax.text(params.rho, 0.05, f" this study: $\\rho$ = {params.rho:g}", transform=ax.get_xaxis_transform(), ha="left",
            fontsize=8.5, color=COLORS["ink_secondary"])
    ax.set(xlabel="$\\rho$", ylabel="largest real part of eigenvalues at C$\\pm$")
    ax.set_title("(b) C+ and C- stop attracting at $\\rho_H$ (Hopf bifurcation)")
    save_figure(fig, figures_dir, "fig01_lorenz_fixed_points")

    metrics = {
        "experiment": "02_lorenz_jacobian",
        "params": dataclasses.asdict(params),
        "jacobian_check": check.summary(),
        "control_sign_error": control.summary(),
        "divergence": lorenz_divergence(params),
        "fixed_points": fixed_points,
        "hopf_rho": {"root_of_growth_rate": rho_hopf, "closed_form": rho_hopf_closed},
        "checks": checks.records,
        "all_checks_passed": checks.all_passed,
    }
    save_json(metrics, args.out / "metrics.json")
    write_run_record(args.out, {key: config[key] for key in ("lorenz", "lorenz_jacobian_check")}, REPO_ROOT,
                     time.perf_counter() - start)

    print(f"Results written to {args.out}")
    print(checks.report())
    for method, entry in check.summary()["methods"].items():
        print(f"Jacobian vs {method:18s}: max relative error {entry['max_relative_error']:.2e}"
              f" (tolerance {entry['tolerance']:.0e}); control with sign error:"
              f" {control.errors[method].max():.2e}")
    for name, entry in fixed_points.items():
        values = ", ".join(f"{ev['real']:.4f}{ev['imag']:+.4f}i" for ev in entry["eigenvalues"])
        print(f"{name:6s} at {np.round(entry['state'], 4)}: eigenvalues {values}")
    print(f"trace J = {lorenz_divergence(params):.6f};  rho_H = {rho_hopf:.6f} (closed form {rho_hopf_closed:.6f})")
    if not check.passed:
        print("Jacobian check FAILED: do not proceed to later phases.")
    return 0 if checks.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
