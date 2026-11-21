# Interpolation study: method and interpretation

This is an exploratory follow-up to Section 4 of `Project Report - Differential Equations -2.pdf`. The original result in `docs/results.md` tested residual direction at one time. This study asks whether reconstructing a sparsely sampled residual and averaging its magnitude over a preceding time interval improves prediction of **future total trajectory error**.

## Inputs and reconstruction

Use the 40 baseline PINNs from eight Lorenz-63 initial-condition windows and five seeds each. Each regenerated run provides the fine grid `t` (spacing 0.002), trajectory error `e(t)`, and automatically differentiated residual vector `r(t)`. Keep only residual samples spaced 0.05 apart, including the endpoint. Reconstruct the vector at fine-grid times by either piecewise linear interpolation or last-sample hold, then take its Euclidean norm. The unthinned dense residual is a measurement benchmark.

For each method, calculate the trapezoidal average of `||r||` on `[t-w,t]` with `w=0.1` and `0.25`. Evaluate predictions **only at sampled knots** with a complete preceding interval and future target inside the PINN window. Linear interpolation then uses only samples observed by prediction time. At intermediate times, piecewise linear interpolation would use the next knot; those times are excluded from the predictive test.

## Two questions kept separate

1. **Reconstruction fidelity:** mean absolute difference between a coarse-reconstructed trailing average and the dense-residual trailing average, divided by the mean dense-residual average, reported per run and averaged over runs. Linear interpolation can win here even if it has no value for predicting error.
2. **Predictive value:** predict `log ||e(t+Δ)||` for `Δ ∈ {0.05, 0.1, 0.25, 0.5}`. Baseline: `log ||r_linear(t)||` plus normalized time. Each candidate adds one log trailing average (linear, hold, or dense) to that same baseline. Fit ordinary least squares on seven initial-condition windows and predict the eighth; repeat for all eight. Report pooled out-of-fold `R²`, mean absolute log error, and the gain in `R²` relative to the baseline. Intervals resample the eight initial-condition windows and keep all five seeds and times from each selected window together.

All candidates use exactly the same prediction times. The dense candidate indicates whether the trailing average contains useful information when sampling error is removed; it is not a deployable sparse-sampling method. The hold candidate tests whether any gain truly needs linear interpolation rather than simple temporal averaging.
The decisive *interpolation-specific* comparison is the paired held-out `R²(linear) - R²(hold)` at each width, with a window-level bootstrap interval. Separate gains over the pointwise baseline cannot answer that comparison on their own.
The output also records the fraction of baseline squared error removed within each held-out window and the number of windows with a positive reduction, so a pooled gain can be checked for dependence on one initial condition.

## How to run and report it

The raw `.npz` files are ignored by git and were absent from this checkout. The current rebuild uses `results/interpolation_rebuild/reference/` and `results/interpolation_rebuild/pinn_runs/`, leaving the published summaries untouched. Once all 40 runs have completed:

```bash
python experiments/09_residual_interpolation.py \
  --runs results/interpolation_rebuild/pinn_runs \
  --out results/interpolation_rebuild/interpolation_sampled_knots
```

The completed output is `results/interpolation_rebuild/interpolation_sampled_knots/metrics.json`. The 40-run completion check, fidelity values, baseline and candidate scores, intervals, design correction, and interpretation are recorded in `docs/interpolation_progress.md`. These widths and models were not part of the original preset decision rule. No directional training loss or Turinici time-weighted training is tested here.
The bootstrap resamples the already computed out-of-fold predictions by initial-condition window; it does not refit the regression for every bootstrap draw. With eight windows and several explored comparisons, the intervals are descriptive rather than confirmatory.
