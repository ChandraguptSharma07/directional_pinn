# Frozen independent validation plan

*Recorded before seeing any results from the new reference windows, 22 November 2026.*

## Purpose

The earlier sequential PINN was developed and evaluated on Lorenz windows
derived from long-run seed state `[1, 1, 1]`. The next check uses a new
long-run seed state `[7, 8, 9]`, with the same transient, 20-unit gap,
solver tolerances, reference checks, model architecture, and analysis code.
No model setting is tuned on the new windows. This is an independent
initial-condition validation within Lorenz-63, not an independent dynamical
system.

## Frozen design

- Reference: `config/independent_validation.yaml`, differing from
  `config/default.yaml` only in `lorenz_reference.seed_state`. Nine windows
  will be generated. Window 0 remains unused; windows 1–8 are the new
  evaluation set.
- Model: the existing `experiments/13_piecewise_pinn_pilot.py` with eight
  segments of length 0.25, four 64-unit tanh layers, hard initial state,
  4,000 Adam steps and 256 stratified collocation points per segment,
  fixed seeds 0–4. Each segment uses the previous predicted endpoint;
  no reference state from inside the window enters training.
- Model-quality outcomes: report all 40 run-level median errors, maxima,
  fraction of the two-unit window with error below 1, and the count below
  error 1 throughout. No run will be dropped because it performs poorly.
- Direction test: use the existing five horizons, the original
  leave-one-window-out and control logic, and exclude exact positive
  multiples of 0.25 from predictor samples because the piecewise derivative
  has two-sided values at joins. Report both candidate directions and the
  oracle full-propagator feature. The original four-part decision rule is
  applied descriptively; this model design was developed after the first
  test, so the new result is not retroactively the original preregistration.
- Uncertainty: identify the eight distinct windows as the independent
  initial-condition units. Five seeds within one window are repeated
  trainings, not five independent trajectories. The existing run-bootstrap
  intervals will be labeled exploratory; a window-level sensitivity will
  be reported separately if feasible.
  `experiments/20_independent_window_analysis.py` was written before model
  results were read; it retains the original held-out predictions and
  resamples whole initial-condition windows for the main alignment and
  oracle gains, with residual-only and residual-plus-time baselines.

## What constitutes a conclusion

The *original* pointwise one-vector hypothesis already has a valid
``not supported in the studied setting'' conclusion. The new round asks
whether the improved model's two-unit accuracy and the repeated negative
directional result transfer to new initial conditions. Whatever the
outcome, the full distribution and failed windows will be presented.
The oracle result can motivate a practical residual-history study but
cannot itself be called a reference-free predictor.

This plan intentionally sets no performance threshold chosen to reproduce
the previous 35/40 success count. The observed accuracy rate and its
window-to-window variation will be reported directly.
