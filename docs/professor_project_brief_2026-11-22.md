# Professor briefing: goal, hypothesis, results, and next conclusion

*Project status as of 22 November 2026. Written in third person for a supervisor presentation.*

## A 90-second explanation

The project began with a question raised by physics-informed neural networks
(PINNs) on chaotic differential equations: **can the direction of a PINN's
equation residual indicate its later trajectory error better than the size of
that residual alone?** The submitted report used Turinici's paper on
time-weighted PINN training as a reference, but the project's own hypothesis
was different. It predicted that two equally large residuals would have
different consequences if one pointed along an expanding direction of the
Lorenz dynamics and the other did not.

The team tested this on Lorenz-63 using accurate numerical reference
trajectories, tangent dynamics, and 40 PINN runs. The planned held-out test
did **not** support the simple pointwise directional hypothesis. Some
within-trajectory associations appeared, but they did not generalize to a
held-out initial condition or clearly beat controls. The original PINNs also
struggled to represent trajectories longer than one time unit, limiting the
study.

The project then improved the model using eight consecutive short PINNs,
each starting from the previous model's predicted endpoint. This produced
accurate two-unit trajectories in **35 of 40 runs**; all five failures were
late divergences on one initial-condition window. Repeating the direction
test on the improved models again found no reliable benefit from alignment
with a single expanding direction. A broader, reference-based quantity that
propagates the *entire residual vector* did predict better at longer horizons.
That result is a mechanistic clue, not yet a practical error indicator.

The project can therefore report a defensible **negative conclusion about
the original pointwise hypothesis in the tested setting** and a substantial
model improvement. A stronger, generalizable conclusion requires a new set
of untouched initial conditions and a predictor that uses only information
available from the PINN, without the true reference trajectory.

## 1. Original goal and hypothesis

**Goal.** Determine whether residual direction adds useful information for
predicting future error of a PINN solving a chaotic initial-value problem.
If it did, that information could eventually guide diagnostics or training.
The project did not assume that a new training loss would work; its plan
required evidence for the diagnostic claim first.

**Initial hypothesis.** For residual vector
$r(t)=\dot{\hat x}(t)-f(\hat x(t))$, the future error should be larger when
$r(t)$ has a greater component along a direction that the dynamics expands.
The tested directions were the backward Lyapunov vector $b_1(t)$ and the
leading finite-time singular vector $s_1(t;\Delta)$. The measurable claim
was that alignment with either direction would improve held-out prediction
of $\|\hat x(t+\Delta)-x(t+\Delta)\|$ beyond residual magnitude $\|r(t)\|$.
The [analysis plan](analysis_plan.md) fixed five horizons, controls, and a
decision rule before the original main analysis.

**Relationship to the reference paper.** Turinici's
[arXiv:2407.21642v2](2407.21642v2.pdf) studies *time weighting of residuals
during PINN training*. This project studies *residual orientation as an
error indicator after training*. The submitted
[project report](Project%20Report%20-%20Differential%20Equations%20-2.pdf)
connects these topics, but one result must not be presented as a test of the
other paper's training method.

## 2. What was tested and what failed

The original experiment used Lorenz-63, a high-accuracy DOP853 reference,
the analytic Jacobian, tangent propagators, Lyapunov and singular vectors,
and 40 standard PINNs from eight initial-condition windows and five seeds.
Prediction was evaluated on a **held-out initial-condition window**, with
random and time-shuffled directions as controls. The original PINNs could be
studied reliably only on one-unit windows.

**Observed result:** Neither $b_1$ nor $s_1$ satisfied the fixed decision
rule at any usable horizon. Some directional associations appeared within
runs, but similar effects appeared for shuffled or neutral directions, and
the main held-out prediction gain was not reliable. The finding is **“not
supported for the tested pointwise predictors,”** rather than proof that
direction never matters. The [original results](results.md) preserve the
complete decision and sensitivity analyses.

**Possible explanations, not proven causes:**

1. The trained residuals were already strongly aligned with $s_1$ (median
   alignment about 0.96 at horizon 0.25), leaving little variation for that
   feature to explain.
2. Total future error contains error accumulated from earlier residuals.
   A single residual measured at time $t$ cannot summarize that history.
3. A global PINN over two time units could produce a small local residual
   while drifting onto an incorrect Lorenz trajectory. That model-quality
   limit restricted the first test to one-unit windows.

The project also explored interpolation and trailing averages of recent
residuals. Those exploratory diagnostics suggested that residual *history*
may be more useful than a pointwise quantity, but they did not change the
original hypothesis verdict or become the current main claim.

## 3. What changed in the model

An exact initial-condition form, stronger early-residual weighting, and
L-BFGS refinement were tested first. Exact initialization improved the
two-unit pilot but still left median trajectory error near 6; the later
adjustments gave little additional benefit. The effective change was a
**sequential time-domain decomposition PINN**: eight separate networks solve
successive 0.25-unit intervals, and each later network starts from the
previous network's *predicted* endpoint. No true interior reference state is
used in training.

On the pilot initial condition, median two-unit error fell from 6.052 for
the single hard-initial-condition PINN to 0.0259 for the sequential model.
With settings fixed, 35/40 runs across eight windows and five seeds stayed
below error 1 throughout two units. The median of run-level median errors
was 0.0433. All five failures occurred late on window 1. This is a strong
model-quality improvement, though the two setups differ in architecture
and training budget, and the new model is not uniformly reliable. The
[research update](piecewise_pinn_research_update.pdf) shows the trajectories
and the full 40-run distribution.

## 4. Where the research stands now

The original held-out direction analysis was repeated on all 40 improved
models, excluding exact segment joins where the derivative has two-sided
values. Neither one-vector alignment passed the original rule. At horizons
0.25, 0.5, and 1.0, an **oracle full-amplification** feature using
$\|\Phi(t+\Delta,t)r(t)\|$ added held-out $R^2$ of approximately +0.051,
+0.195, and +0.307 beyond residual size. Its gains remained positive after
adding elapsed time and its square to the baseline (+0.046, +0.191, +0.360)
and after removing the late-failing window 1. The full tangent propagator
$\Phi$ was calculated along the known reference trajectory, so this is an
**explanation-oriented oracle**, not a deployable prediction method.

The defensible present conclusion is narrow: **for these Lorenz PINNs,
pointwise alignment with one expanding direction is not a reliable
held-out predictor of total future error.** More complete dynamical
amplification appears relevant, and the improved PINN supplies a better
trajectory test bed. This does not establish why the one-vector predictor
fails, nor does it validate a new practical predictor. The sequential
model was chosen after the original result, so its repeat analysis is
exploratory rather than an independent preregistered test.

## 5. When can a conclusion be presented?

**Now:** The professor can be shown a complete project conclusion about
the *original* hypothesis, the new model's measured accuracy, its failure
case, and the oracle mechanism. The [three-page research update](piecewise_pinn_research_update.pdf)
and [chronological run record](model_improvement_progress.md) already
support that presentation.

**After one frozen independent validation round:** A stronger conclusion
about the new model and residual-direction result requires initial-condition
windows that played no part in model design or inspection. The settings,
horizons, exclusions, controls, and window-level uncertainty method should
be fixed before those runs. All failures must remain in the analysis. This
is the next experimental milestone, not a missing explanation that can be
filled in by writing alone.

**For a positive practical predictor claim:** The project must implement a
causal statistic from past PINN residuals, compare it with residual size,
elapsed time, and simple trailing-history controls, and validate it on those
untouched windows. Its input cannot include the true trajectory or oracle
tangent propagator. The piecewise Duhamel quadrature also needs correction
at segment joins before it is used as mechanistic evidence.

The computation for another 40-run round was feasible on the current
machine, but the date of a stronger conclusion depends on its outcome.
If it reproduces the current pattern, the paper's defensible claim is a
carefully bounded negative result plus a model-quality advance and an
oracle-to-practical research direction. If it does not, the conclusion must
be revised rather than declared on a timetable.

## Suggested answer to a direct question from the professor

> **“Did the initial hypothesis fail?”** The planned test did not support
> the hypothesis that one pointwise expanding-direction alignment predicts
> future Lorenz PINN error beyond residual magnitude. The improved PINN
> repeated that pattern. The result is specific to this tested predictor and
> setting; the full-vector oracle result suggests that dynamical
> amplification still matters.

> **“What is the contribution now?”** The project has a reproducible
> negative test, a sequential PINN that extends accurate Lorenz trajectories
> to two units in 35/40 runs, and evidence separating a weak one-vector
> statistic from a stronger full-propagator oracle. The next contribution
> under test is a causal, reference-free residual-history indicator.
