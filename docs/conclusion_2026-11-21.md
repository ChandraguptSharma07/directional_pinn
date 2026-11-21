# Project conclusion and continuation decision (21 November 2026)

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
4. The project has now tested sparse linear interpolation plus a trailing residual average as an **exploratory follow-up**; see `docs/interpolation_progress.md`. It has not tested Gaussian-process smoothing, Duhamel-integrated predictors as held-out regressors, or Turinici-style time weighting in training. The report's suggestion that weighting may resolve the cause is a hypothesis, not a result.

## Decision for the next study

Close the original question with the negative conclusion above. The next question is whether **time-averaged residual information** improves held-out prediction. The implemented first study compares linearly interpolated trailing residual averages with a sample-and-hold control and the original dense residual, using a pointwise-residual-plus-time baseline. A later mechanistic study could test the error created inside the horizon and compare a Duhamel-weighted integral with the oracle propagator. Any outcome belongs to new exploratory work, separate from the preset decision rule.

The interpolation study is implemented in `experiments/09_residual_interpolation.py`. It compares piecewise linear reconstruction from residual samples 0.05 apart with sample-and-hold and the original dense residual, then tests trailing averages of widths 0.1 and 0.25. Predictive evaluation uses only sampled knots, so interpolation uses no residual from after the prediction time. All models use the same times, a pointwise-residual-plus-time baseline, and leave-one-initial-condition-window-out prediction. Window-level bootstrap intervals keep all seeds from one initial condition together. Results and design corrections are recorded in `docs/interpolation_progress.md`.

With all 40 PINNs rebuilt, a width-0.25 linear trailing average improves held-out `R²` over that baseline by +0.474, +0.472, and +0.312 at future horizons 0.05, 0.1, and 0.25. Its direct gain over sample-and-hold is smaller, +0.058, +0.055, and +0.052. At horizon 0.5, improvement over baseline is uncertain. These are exploratory associations on a restricted subset of times, not a revision of the original preset decision.

Running that study required regenerating the untracked `trajectory.npz` and `residuals.npz` arrays; the committed JSON summaries alone could not support a new time-series analysis. Retraining with Turinici's weights is a separate controlled experiment requiring the same seeds, compute budget, and accuracy metrics as the baseline. That experiment is not needed to state the original conclusion.

## Evidence

- `docs/analysis_plan.md`: fixed hypothesis, held-out design, controls, and decision rule.
- `results/analysis/metrics.json`: 40-run main analysis and reported intervals.
- `results/analysis_lambda10/metrics.json`, `results/analysis_float64/metrics.json`: sensitivity analyses.
- `docs/results.md`: detailed interpretation, figures, and limitations.
- `docs/phase4_pinn.md`: pilot, one-unit window, and initial-condition-weight deviation.
- `docs/2407.21642v2.pdf`, Sections 2–4: reference paper and the assumptions behind its weighting formula.
- `docs/Project Report - Differential Equations -2.pdf`: submitted project report.
