# Project conclusion and continuation decision (21 November 2026)

> **22 November continuation:** A consecutive hard-IC PINN stayed below
> trajectory error 1 over a two-unit Lorenz window in 35/40 development
> runs. Its settings were then frozen and tested on fresh initial conditions:
> **30/40** runs stayed below error 1 throughout and **32/40** for at least
> 90% of the window. Neither one-vector alignment passed the direction rule
> on the fresh runs; an oracle full-propagator feature added held-out
> information at longer horizons. This strengthens a bounded Lorenz-specific
> negative result while showing that the model is not uniformly reliable.
> See the [updated professor briefing](professor_project_brief_2026-11-22.pdf),
> [validation plan](independent_validation_plan.md), and
> [validation log](validation_progress_2026-11-22.md).

> **Later 22 November adaptive-weighting test:** The paper's state-based
> time-weighting rule was ported to our PyTorch PINN and compared with uniform
> weighting on the same fixed grid, seed, architecture, and budget. The global
> two-unit pilot remained inaccurate with either weighting. In eight seed-0
> sequential windows, the equation-based adaptive scale lowered median error
> in 1/8; both it and uniform weighting stayed below error 1 throughout in
> 7/8. The author's literal code time scale lowered median error in 0/8 and
> kept 6/8 below error 1. See the
> [run log](adaptive_weighting_progress_2026-11-22.md) and
> [updated final report](final_project_report.pdf). The chronological text below
> describes the state before this later test.

## What this project actually tests

The reference paper, Gabriel Turinici's *Regime-Aware Time Weighting for Physics-Informed Neural Networks* ([arXiv:2407.21642v2](https://arxiv.org/abs/2407.21642v2)), asks how to weight **when** a PINN violates a differential equation. It derives a scalar time weight from a worst-case final-error bound, then estimates the needed growth rate from the evolving PINN. Its Lorenz experiment changes the training loss.

This repository asks a different, narrower question: for a **standard**, already-trained PINN solving Lorenz-63, does the **direction** of a residual at one time predict later trajectory error beyond the residual's size? It computes expanding directions from a high-accuracy reference solution and tests two alignments, with controls and a decision rule recorded in `docs/analysis_plan.md` before the main analysis. The local PDF `Project Report - Differential Equations -2.pdf` summarizes the paper, the project's negative result, and possible follow-up work.

## Conclusion supported by completed experiments

**The pointwise directional-alignment hypothesis is not supported for the studied PINNs.** In 40 main runs (8 initial conditions × 5 seeds), neither alignment with the backward Lyapunov vector \(b_1\) nor with the leading finite-time singular vector \(s_1\) passes the preset rule at any usable horizon. Leave-one-window-out gain in \(R^2\) over a model using log residual size alone ranges from −0.008 to +0.020 for \(b_1\) and from +0.017 to +0.046 for \(s_1\). No horizon meets both the minimum gain of 0.02 and a positive 95% interval for either direction. The result also holds in 40 runs with initial-condition weight 10 and in the smaller float64 check (8 runs).

There is a limited positive observation: the **oracle** quantity \(\|\Phi(t+\Delta,t)r(t)\|\), which accounts for all directions and their finite-time growth, adds approximately 0.066 and 0.055 to held-out \(R^2\) at horizons 0.25 and 0.5 in the main set, with positive reported bootstrap intervals. This was a secondary analysis, and it uses a propagator calculated along the known reference trajectory. It is evidence that full dynamical amplification matters in this setting; it is not evidence that a practical predictor or a new PINN training loss has succeeded.

The standard PINNs could only be studied over windows of length 1. The leading singular-vector alignment is already high for most residuals (median 0.96 at horizon 0.25), limiting the variation available to a one-direction predictor. Inherited error from earlier residuals also makes instantaneous residual size a weak target for near-future total error. These observations help interpret the result, but the data do **not** establish that the optimizer deliberately removes contracting components or that either mechanism is the unique cause of failure.

## Corrections to carry into the final presentation

1. Turinici's \(\rho_{\mathrm{opt}}(t)\propto\exp[-\int_0^t\lambda(s)\,ds]\) is optimal **for the paper's stated minimax problem**, which holds a weighted residual-loss budget fixed and assumes the initial condition is satisfied exactly. It is not a universal guarantee of minimum final error for any finite neural-network training run. Our PINN instead uses a soft initial-condition penalty.
2. The paper's scalar \(\lambda(t)\) bounds the one-sided error growth in Proposition 1; the practical estimator in Eq. (17) substitutes the current PINN state and zero for the unknown exact state. It should not be described as a precisely measured Lyapunov exponent. A scalar bound discards direction because it is a bound, not because direction is physically irrelevant.
3. The error equation is exact as \(\dot e=f(\hat x)-f(x)+r\). Replacing \(f(\hat x)-f(x)\) by \(J(x)e\) gives the **linearized** Duhamel reconstruction used here. It is an approximation for nonlinear Lorenz-63; its reported median relative mismatch is about 6% across runs, despite some runs having large errors.
4. The project has now tested sparse linear interpolation plus a trailing residual average as an **exploratory follow-up**; see `docs/interpolation_progress.md`. It has not tested Gaussian-process smoothing or Duhamel-integrated predictors as held-out regressors. The later adaptive-weighting test is documented above; it did not establish an improvement in our setup.

## Decision for the next study

Close the original question with the negative conclusion above. The follow-up asks whether **time-averaged residual information** improves held-out prediction. The implemented study compares linearly interpolated trailing residual averages with sample-and-hold, plain sampled-knot averaging, and the original dense residual, using a pointwise-residual-plus-time baseline. A later mechanistic study could test the error created inside the horizon and compare a Duhamel-weighted integral with the oracle propagator. This follow-up is exploratory work, separate from the preset decision rule.

The interpolation study is implemented in `experiments/09_residual_interpolation.py`. It compares piecewise linear reconstruction from residual samples 0.05 apart with sample-and-hold and the original dense residual, then tests trailing averages of widths 0.1 and 0.25. Predictive evaluation uses only sampled knots, so interpolation uses no residual from after the prediction time. All models use the same times, a pointwise-residual-plus-time baseline, and leave-one-initial-condition-window-out prediction. Window-level bootstrap intervals keep all seeds from one initial condition together. Results and design corrections are recorded in `docs/interpolation_progress.md`.

With all 40 PINNs rebuilt, a width-0.25 linear trailing average improves held-out `R²` over that baseline by +0.474, +0.472, and +0.312 at future horizons 0.05, 0.1, and 0.25. Its direct gain over sample-and-hold is smaller, +0.058, +0.055, and +0.052. At horizon 0.5, improvement over baseline is uncertain. These are exploratory associations on a restricted subset of times, not a revision of the original preset decision.

A post-result control that simply averages sampled residual magnitudes, including the current sample, gives similar gains to sample-and-hold. Linear interpolation improves held-out `R²` over this simpler control by about +0.05 to +0.06 at horizons 0.1 and 0.25 for a width-0.25 average. This is a modest exploratory interpolation effect; the stronger finding is that averaging recent residual magnitudes helps at short horizons. Details and the retained invalid look-ahead run are in `docs/interpolation_progress.md`.

Running that study required regenerating the untracked `trajectory.npz` and `residuals.npz` arrays; the committed JSON summaries alone could not support a new time-series analysis. The subsequent controlled port of Turinici's adaptive rule uses matched seeds, fixed grids, compute budgets, and trajectory-accuracy metrics; its limited one-seed results are in `docs/adaptive_weighting_progress_2026-11-22.md`. It does not change the original directional conclusion.

## Evidence

- `docs/analysis_plan.md`: fixed hypothesis, held-out design, controls, and decision rule.
- `results/analysis/metrics.json`: 40-run main analysis and reported intervals.
- `results/analysis_lambda10/metrics.json`, `results/analysis_float64/metrics.json`: sensitivity analyses.
- `docs/results.md`: detailed interpretation, figures, and limitations.
- `docs/phase4_pinn.md`: pilot, one-unit window, and initial-condition-weight deviation.
- `docs/2407.21642v2.pdf`, Sections 2–4: reference paper and the assumptions behind its weighting formula.
- `docs/Project Report - Differential Equations -2.pdf`: submitted project report.
