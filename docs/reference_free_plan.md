# Reference-free directions: plan

*Written on 22 November 2026. This is before any reference-free direction has
been computed, and before any PINN on the fresh starting points below has been
trained.*

## Why

The dial study (`docs/dial_study_results.md`) found that a residual's direction
predicts the error it creates when the stretching gap is large. That was
verdict "partly supported" under its rule: about +0.07 to +0.13 in held-out R²
at rate differences 17–65, and nothing at Lorenz-like gaps.

That study used **oracle** directions. s_1 was computed from the tangent
propagator along the exact reference solution, which a real PINN user does not
have. This plan tests whether the same information can be obtained from the
PINN alone.

## Reference-free direction estimates

Both estimates use only the equations, through the Jacobian J, and the trained
PINN's own output x_hat(t). Neither uses the reference solution.

- **P, the PINN-path propagator (primary).** Integrate the tangent equation
  dPhi/dt = J(x_hat(t)) Phi along the PINN's own trajectory over [t, t+dt].
  The PINN state between grid points comes from a cubic spline through the PINN
  outputs on the evaluation grid (spacing 0.002). The estimate s1_P(t; dt) is
  the leading right singular vector of that propagator.
- **F, the frozen Jacobian (secondary).** s1_F(t; dt) is the leading right
  singular vector of expm(J(x_hat(t)) dt). It uses only the state at time t.

For the linear saddle, J is constant, so s1_P = s1_F = s_1 exactly. Only the
nonlinear Davis–Skodje family (and Lorenz) can tell whether the estimate is
good. The saddle settings serve only to replicate the oracle result and as a
low-gap control.

## Data

**Fresh starting points.** Windows 9–16 are new and fixed now. The PINN
settings, out_mean/out_scale and analysis are identical to the dial study
(including the soft initial condition with lambda_IC = 100, 20,000 Adam
iterations, seeds 0–4 and T = 2).

| window | saddle (c, d) | Davis–Skodje (y1(0), offset) |
|---:|---|---|
| 9 | (1.1, −0.4) | (0.75, −0.15) |
| 10 | (0.75, 0.6) | (1.25, 0.25) |
| 11 | (1.25, −0.9) | (1.75, −0.25) |
| 12 | (0.65, 0.3) | (2.25, 0.1) |
| 13 | (0.95, −0.6) | (2.75, −0.1) |
| 14 | (1.35, 0.9) | (3.25, 0.2) |
| 15 | (0.85, −0.15) | (3.75, −0.2) |
| 16 | (1.15, 0.45) | (0.6, 0.3) |

**Settings trained on the fresh windows:** saddle kappa = 1 (low-gap control),
saddle kappa = 64, Davis–Skodje gamma = 18 and gamma = 66. That is 4 × 40 = 160
runs, on the GPU as stacked models.

**Development data, descriptive only.** The 320 dial-study runs (windows 1–8)
and the October Lorenz runs are analysed with the same code. They inform
nothing in this plan, because P, F and the rules below are fixed now.

## Test

The samples, horizons, target and baseline are those of the dial study:
- **Samples:** every 0.01 time units.
- **Target:** the log of the created error N at the primary horizon dt = 0.1.
- **Baseline B:** log R, tau and tau².
- **Evaluation:** leave-one-window-out, with 95% intervals from 2000 bootstrap
  resamples of whole windows.

**Models.** Each adds one feature to B, the log of the residual component along
a direction (with the same floor as before):
- M2-oracle uses s_1;
- M2-P uses s1_P;
- M2-F uses s1_F;
- the control uses 20 random directions.

Target N still needs the reference, because it is what we try to predict. Only
the predictors must be reference-free.

**Decision on the fresh windows.**

(A) **Replication of the dial finding.** The oracle direction "works" at kappa =
64, gamma = 18 and gamma = 66 if:
- the gain is at least 0.05;
- its interval lies above 0;
- the coefficient is positive;
- the gain exceeds the random-direction 95th percentile.

It "fails" at kappa = 1 if the gain is below 0.02. The time-shuffle control
is not used, because the dial study showed it is degenerate where s_1 does not
turn.

(B) **Reference-free direction (the question of this plan).** For
Davis–Skodje gamma = 18 and gamma = 66, M2-P "works" if:
- it meets the same four conditions;
- **and** its gain is at least half the oracle gain on the same samples.

**Verdict:**
- "supported" if (B) holds at both settings;
- "partly supported" if at one;
- "not supported" if at neither.

(A) is reported separately.

**Also reported:**
- F, judged by the same rule;
- the angle between s1_P and s_1, and between s1_F and s_1 (median and 90th
  percentile);
- the paired difference between the P and oracle gains, with its interval;
- all horizons;
- the total-error target;
- the development data;
- the Lorenz runs.

## What this cannot show

- **The systems are small.** They are two-dimensional and simple. For the
  saddle, the reference-free direction is trivially exact.
- **A good direction is not a deployable error estimate.** The created-error
  target is defined with the reference.

A positive result would show that the useful direction can be computed without
the reference in this nonlinear stiff example. It would not show that any
error bound or estimator is ready.
