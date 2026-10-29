"""Phase 4 pilot: choose the PINN window length and initial-condition weight on window 0.

Run from the repository root (after experiments/03_lorenz_reference.py):

    python experiments/05_pinn_pilot.py

Window 0 is never used in the analysis. The selection rule is fixed before the
runs (and written in config/default.yaml and docs/phase4_pinn.md):

1. lambda_IC: among the candidates, the one with the smallest median PINN error
   |x_hat(t) - x(t)| over the window, averaged over the candidate window lengths.
2. Window length T: the longest candidate for which the PINN with the chosen
   lambda_IC keeps the error below 1 (about 5 % of the attractor's size) for at
   least 90 % of the window, so that most of the data stay in the regime where
   the tangent (linear) error picture applies.

The rule looks only at PINN accuracy, never at residual directions.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # lets `python experiments/05_pinn_pilot.py` import src

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.plots import COLORS, apply_style, ordinal_colors, save_figure
from src.pinn.runs import run_job
from src.utils.reproducibility import load_config, save_json, write_run_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 pilot: choose PINN settings on window 0.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default.yaml")
    parser.add_argument("--reference", type=Path, default=REPO_ROOT / "results" / "lorenz_reference")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "pinn_pilot")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    config = load_config(args.config)
    cfg = config["pinn_pilot"]
    window = int(cfg["window"])
    reference_file = args.reference / "data" / f"window_{window}.npz"
    x0 = np.load(reference_file)["x0"].tolist()
    jobs = []
    for t_final, lambda_ic in itertools.product(cfg["window_lengths"], cfg["lambda_ic"]):
        train = {**config["pinn"], "t_final": float(t_final), "lambda_ic": float(lambda_ic), "x0": x0,
                 "seed": int(cfg["seed"])}
        jobs.append({
            "run_dir": str(args.out / "runs" / f"T{t_final:g}_lambda{lambda_ic:g}"),
            "window": window,
            "reference_file": str(reference_file),
            "params": config["lorenz"]["params"],
            "train": train,
        })
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = list(pool.map(run_job, jobs))

    lengths = sorted({s["t_final"] for s in summaries})
    lambdas = sorted({s["lambda_ic"] for s in summaries})
    table = {(s["t_final"], s["lambda_ic"]): s for s in summaries}
    score = {lam: np.mean([table[T, lam]["median_error"] for T in lengths]) for lam in lambdas}
    chosen_lambda = min(lambdas, key=lambda lam: score[lam])
    acceptable = [T for T in lengths if table[T, chosen_lambda]["fraction_error_below_1"] >= 0.9]
    chosen_length = max(acceptable) if acceptable else min(lengths)

    apply_style()
    fig, axes = plt.subplots(1, len(lambdas), figsize=(4.4 * len(lambdas), 3.8), constrained_layout=True,
                             sharey=True)
    for ax, lam in zip(np.atleast_1d(axes), lambdas):
        for T, color in zip(lengths, ordinal_colors(len(lengths))):
            data = np.load(Path(args.out / "runs" / f"T{T:g}_lambda{lam:g}" / "trajectory.npz"))
            ax.plot(data["t"], np.linalg.norm(data["error"], axis=1), color=color, label=f"T = {T:g}")
        ax.axhline(1.0, color=COLORS["muted"], lw=1.0)
        ax.set_yscale("log")
        ax.set(xlabel="t", title=f"$\\lambda_{{IC}}$ = {lam:g}" + (" (chosen)" if lam == chosen_lambda else ""))
        ax.legend(loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("PINN error $\\Vert\\hat x(t) - x(t)\\Vert$")
    save_figure(fig, args.out / "figures", "fig01_pilot_errors")

    metrics = {
        "experiment": "05_pinn_pilot",
        "window": window,
        "runs": summaries,
        "lambda_score_mean_median_error": {f"{lam:g}": score[lam] for lam in lambdas},
        "chosen_lambda_ic": chosen_lambda,
        "acceptable_window_lengths": acceptable,
        "chosen_window_length": chosen_length,
    }
    save_json(metrics, args.out / "metrics.json")
    write_run_record(args.out, {key: config[key] for key in ("lorenz", "pinn", "pinn_pilot")}, REPO_ROOT,
                     time.perf_counter() - start)
    print(f"Results written to {args.out}")
    for s in summaries:
        print(f"T={s['t_final']:g} lambda={s['lambda_ic']:g}: median error {s['median_error']:.3g}, max {s['max_error']:.3g},"
              f" error<1 on {100 * s['fraction_error_below_1']:.0f}% of window, initial error {s['initial_error']:.2g},"
              f" median |r| {s['median_residual']:.3g}, {s['runtime_seconds']:.0f}s")
    print(f"chosen lambda_IC = {chosen_lambda:g}, chosen window length T = {chosen_length:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
