# Dial study: when can the direction of a PINN residual predict its error?

*Plan written on 22 November 2026, before any PINN in this study was trained.
It is a new, separate hypothesis; it does not change the verdict of the
original Lorenz study (`docs/analysis_plan.md`, `docs/results.md`).*

## 1. Why this study

The original question was whether the direction of a PINN residual, relative
to the most expanding direction of the dynamics, predicts later trajectory
error beyond the residual's size. For Lorenz-63 the answer was "not
supported", on the planned 40-run set and on a fresh 40-run set.

The error equation itself says that direction *can* matter. A residual acts as
a stream of small pushes on the solution, and the dynamics stretch or shrink
each push according to its direction (Phase 0). So the useful question is not
"does direction matter?" but "when is the effect big enough to be seen?". Three
conditions are needed:

1. **A large stretching gap** over the horizon, so that pointing in one
   direction rather than another changes the outcome a lot.
2. **Residual directions that vary**, so that direction can distinguish one
   case from another.
3. **A target that the current residual actually creates.** Error inherited
   from earlier residuals cannot be explained by the residual at time t.

An exploratory check of the October Lorenz data (not part of any earlier plan)
gave the following numbers:

- **The gap is small.** The ratio sigma_1/sigma_2 of the two largest singular
  values of Phi(t+dt, t) has a median of 1.4 at dt = 0.05, 1.8 at dt = 0.1,
  3.0 at dt = 0.25 and 7.2 at dt = 0.5. Over the same samples, the future error
  varies by a factor of about 3000 (90th/10th percentile).
- **Residuals are already aligned.** The median alignment with s_1 is
  0.87–0.96.
- **The residual direction changes quickly.** The mean signed cosine between
  r(t) and r(t+lag) is 0.52 at lag 0.05, 0.51 at lag 0.1, 0.15 at lag 0.25
  and −0.02 at lag 0.5.

So in Lorenz the residual direction keeps its meaning for only about 0.1 time
units, and over that horizon the dynamics hardly separate directions. A test of
the idea needs systems with a large gap at **short** horizons. A strongly
contracting ("stiff") direction gives exactly that. A residual along a stiff
direction is erased within about 1/kappa time units, while the same residual
along a slow or expanding direction persists or grows. Lorenz has a contracting
direction (exponent −14.6), but trained residuals almost never point along it
(median alignment with s_3 is 0.06–0.13).

## 2. Hypothesis

**H_dial.** For PINNs trained with the standard residual loss, the held-out
information that the residual's direction adds about the error it creates
grows with the stretching gap. When the gap is Lorenz-like the gain is
negligible; when the gap is large it is substantial.

The gap is varied by a **dial**: the rate difference lambda_1 − lambda_2
between the leading direction and the other direction. The dial is set to
2, 5, 17 and 65 in two families of two-dimensional systems.

The hypothesis is falsified if direction adds no reliable information even
at the largest gap, or if the gain does not increase with the gap.

## 3. Systems

Both families are two-dimensional and autonomous, and each is used at four
dial settings. Every setting is trained and analysed separately.

**Family L, stiff linear saddle.**
dz/dt = A z, where A = R diag(1, −kappa) R^T and R is the rotation by 30
degrees. The leading (expanding) eigenvector is v_u = (cos 30°, sin 30°), with
rate +1. The stable eigenvector is v_s = (−sin 30°, cos 30°), with rate
−kappa. The settings are kappa ∈ {1, 4, 16, 64}, so the rate difference is
1 + kappa ∈ {2, 5, 17, 65}. The exact solution is
z(t) = c e^t v_u + d e^(−kappa t) v_s, and Phi(t+dt, t) = expm(A dt) is the
same at every time. The rotation makes both network outputs mix both modes.

**Family D, Davis–Skodje slow–fast system** (a standard nonlinear test problem
for slow manifolds):

    dy1/dt = −y1
    dy2/dt = −gamma y2 + ((gamma − 1) y1 + gamma y1²) / (1 + y1)²

The curve y2 = y1/(1 + y1) is an invariant slow manifold. The Jacobian is lower
triangular with eigenvalues −1 (slow) and −gamma (fast). Its off-diagonal term
makes it non-normal, so the singular directions turn as the state moves. The
settings are gamma ∈ {3, 6, 18, 66}, so the rate difference is gamma − 1 ∈
{2, 5, 17, 65}, the same dial values as family L.

**Window.** T = 2 for both families, with initial-condition windows 0–8.
Window 0 is used only for the pilot; windows 1–8 are the analysis set. The
windows are the same for all four settings of a family:

| window | L: (c, d) | D: (y1(0), offset) |
|---:|---|---|
| 0 | (1.0, 0.6) | (1.25, +0.15) |
| 1 | (1.0, 0.5) | (0.5, +0.1) |
| 2 | (0.6, −0.5) | (1.0, −0.1) |
| 3 | (1.4, 0.25) | (1.5, +0.2) |
| 4 | (0.8, −1.0) | (2.0, −0.2) |
| 5 | (1.2, 1.0) | (2.5, +0.3) |
| 6 | (0.7, −0.25) | (3.0, −0.3) |
| 7 | (1.3, −0.75) | (3.5, +0.15) |
| 8 | (0.9, 0.75) | (4.0, −0.15) |

For family D, y2(0) = y1(0)/(1 + y1(0)) + offset. Each start is therefore
slightly off the slow manifold and has a fast transient of rate gamma.

**Lorenz-63 reference point.** The October 40-run set (`results/pinn_runs`,
T = 1) is put through the same analysis code as a low-gap point. It is
descriptive, not part of the decision, because its data have already been
seen.

## 4. Reference solutions and directions

- **Grid.** The grid spacing is h = 0.002 on [0, 2] (1001 times).
- **Family L.** Exact solution. Phi over any horizon is expm(A dt).
- **Family D.** DOP853 with rtol 1e-11 and atol 1e-13. A check against a
  solution with tolerances ten times looser must agree to 1e-8. Step
  propagators come from the existing tangent integrator. Phi(t+dt, t) is the
  product of step propagators.
- **Directions.** For each horizon, s_1(t; dt) is the leading right singular
  vector of Phi(t+dt, t), and sigma_1 ≥ sigma_2 are its singular values. These
  are oracle quantities computed from the reference. The backward Lyapunov
  vector b_1 is not used: these trajectories have no long past to converge on.
- **Integrated propagator.** K(t; dt) = ∫ Phi(t+dt, s) ds over s from t to
  t+dt is computed by the trapezoidal rule. It maps a constant residual to the
  error it creates over the horizon. It is used only for a secondary oracle
  feature and for describing the dial.

## 5. PINNs

The model and training follow the existing code, except that the
right-hand side is now general:

- **Network:** four tanh layers of 64 units, hard initial condition
  x_hat(t) = x0 + (t/T)(g(t) − x0).
- **Training:** Adam, learning rate decaying from 1e-3 to 1e-5, 1024
  stratified collocation points per iteration, no curriculum, no time
  weighting, float32.
- **Evaluation:** float64 on the reference grid.
- **Output scaling:** out_mean is the mean and out_scale is the standard
  deviation (floored at 0.1) of the window-0 reference states of each setting.
  The scaling is the same for every window of that setting.
- **Runs:** seeds 0–4 on windows 1–8, which gives 40 runs per setting and 320
  runs in all. Every run is kept and reported. Runs with non-finite outputs
  are excluded from the samples and listed.

**Amendment made before any training (22 November 2026).** Two settings above
are changed, before any PINN of this study was trained.

1. **Soft initial condition.** The initial condition is enforced by the
   penalty lambda_IC = 100, as in the original Lorenz runs, instead of the
   hard form. With x_hat = x0 + (t/T)(g − x0), a fast transient of rate kappa
   from an offset d forces the raw output g to reach about T · kappa · d near
   t = 0. For kappa = 64 and d = 1 that is about 128, which would make the
   stiff settings mostly a test of representing that spike.
2. **Output scaling from all nine windows.** out_mean and out_scale are the
   per-component mean and standard deviation (floored at 0.1) of the
   reference states of all nine windows of the setting, rather than of
   window 0 alone. For family D, window 8 starts at y1 = 4, far outside
   window 0's range.

The scaling uses only reference solutions, never PINN outcomes.

**Pilot rule (accuracy only, decided now).** Window 0, seed 0, is trained at the
stiffest setting of each family (kappa = 64, gamma = 66) with 20,000
iterations. The pilot passes if the maximum trajectory error is at most 5% of
the maximum reference state norm in the window. If either pilot fails, both
families use 40,000 iterations, with no further tuning. The pilot never
computes direction statistics.

## 6. Samples, targets and features

**Sample times.** Samples are taken every fifth grid point (every 0.01) at
times t with t + dt ≤ T.

**Horizons.** dt ∈ {0.05, 0.1, 0.25, 0.5}. The **primary horizon is
dt = 0.1**, because that is where the Lorenz residual direction still keeps its
meaning.

**Targets.**
- **Primary:** the error created inside the horizon,
  N_dt(t) = ||e(t+dt) − Phi(t+dt, t) e(t)||. It is the part of the future
  error that residuals after time t produce.
- **Secondary:** the total future error E_dt(t) = ||e(t+dt)||.

Both targets are modelled on a log scale, with log(x + 1e-12).

**Features.** These follow the definitions of the original plan:
- R = ||r(t)||;
- R_u = |r · s_1|;
- alignment A = R_u / R;
- time tau = t/T.

**Models.** Each model is an ordinary least-squares fit of the log target.
- B (baseline): log R, tau, tau².
- M2 (**primary direction model**): B + log(R_u + 1e-6 · median R of the run).
  Given log R, this is the log of the alignment. A log form is used because
  the dynamics multiply the residual component along each direction.
- M1 (secondary): B + A, the linear alignment form of the original plan.
- M3 (secondary, oracle): B + log(||Phi(t+dt, t) r|| / R).
- M4 (secondary, oracle): B + log(||K(t; dt) r|| / R).

## 7. Held-out evaluation and uncertainty

Evaluation is leave-one-window-out: fit on seven windows (35 runs) and predict
the eighth, repeated for all eight. The reported quantities are the pooled
out-of-fold R² and the gains Delta R² = R²(model) − R²(B).

Intervals resample the eight windows with replacement (2000 draws), with the
out-of-fold predictions held fixed. They describe the uncertainty across
initial conditions and do not support a confirmatory claim beyond these
systems.

## 8. Controls

- **Random directions.** The same gain as M2 is recomputed with s_1 replaced by
  independent uniformly random unit vectors at every time, 20 draws.
- **Time-shuffled direction (family D only).** The same gain is recomputed with
  s_1 permuted in time within each run, 20 permutations. Family L has a
  constant s_1, so a time shuffle changes nothing there.

## 9. Decision rules

**Per setting.** These rules apply at the primary horizon dt = 0.1 with target
N.

**Direction works** if all of the following hold:
1. Delta R²(M2) ≥ 0.05;
2. its 95% window-bootstrap interval lies above 0;
3. the coefficient of log R_u in the all-data fit of M2 is positive;
4. Delta R²(M2) exceeds the 95th percentile of the random-direction gains,
   and, for family D, of the time-shuffled gains.

**Direction fails** if Delta R²(M2) < 0.02. Anything else is
**inconclusive**.

**Per family.** H_dial is supported in a family if all three hold:
- (a) the largest setting (dial 65) "works";
- (b) the smallest setting (dial 2) "fails";
- (c) Kendall's tau between the dial value and Delta R²(M2) across the four
  settings is at least 0.66, which allows at most one discordant pair.

**Overall.**
- "Supported" if H_dial holds in both families.
- "Partly supported" if it holds in one.
- "Not supported" if it holds in neither.

Secondary targets, horizons, models and the Lorenz reference point are
reported in full but do not change the verdict.

## 10. Descriptive reports

These are reported for every setting:
- the median and 10th/90th percentiles of sigma_1/sigma_2 and of the
  integrated-lever ratio (largest over smallest singular value of K);
- the 10th, 50th and 90th percentiles of the alignment A;
- the residual direction persistence (mean signed cosine of r(t) and r(t+dt));
- the trajectory accuracy (median and maximum error per run);
- the per-window gains.

## 11. What this study cannot show

- **The directions are oracle quantities.** They are computed from the
  reference solution. A positive result would show when direction carries
  information, not that a reference-free predictor exists.
- **The systems are small and two-dimensional.** Family L is linear, so its
  linearized error relation is exact. Family D tests one nonlinear case.
  Neither replaces a study of high-dimensional or chaotic systems.
- **It does not run Phase 9.** The directional training loss from the original
  plan stays conditional on a supported directional finding in the system
  where it would be used.

## 12. Deviations

Any change to this plan after PINN training starts is recorded with its
reason in `docs/dial_study_progress.md`. Results under the original plan are
then also reported.
