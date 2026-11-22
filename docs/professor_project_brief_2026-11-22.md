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
accurate two-unit trajectories in **35/40 development runs**. A frozen
evaluation on **40 new runs from untouched initial conditions** found
**30/40 accurate throughout** and **32/40 accurate for at least 90%** of
the window. Repeating the direction test on the new runs still found no
reliable benefit from a single expanding direction. A broader,
reference-based quantity that propagates the *entire residual vector*
predicted better at longer horizons. That is a mechanistic clue, not yet a
practical error indicator.

The project can therefore report a defensible **negative conclusion about
the original pointwise hypothesis in the tested Lorenz setting**, now
reinforced on fresh initial conditions. The model improvement is useful but
less reliable than the first 35/40 figure suggested. A positive practical
predictor still needs a method that uses only information available from
the PINN, without the true reference trajectory.

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
With settings fixed, **35/40 development runs** across eight windows and
five seeds stayed below error 1 throughout two units; all five failures
occurred late on window 1. Before examining a new set, the model and
analysis settings were frozen in the
[independent validation plan](independent_validation_plan.md). A fresh
long-run seed produced eight new evaluation windows. The high-accuracy
reference passed 36/36 checks, and the tangent calculations passed 31/31.
On these **40 new runs**, 30/40 stayed below error 1 throughout, 32/40
for at least 90% of the window, and the median of run-level median errors
was 0.0551. Failures occurred on windows 1, 3, 6, and 7; all five seeds
failed the full-window criterion on window 1. The model therefore extends
accuracy beyond the original global PINN but is not uniformly reliable.
The [fresh-set figure](figures/independent_validation_40runs.pdf) shows
every window and seed.

## 4. Where the research stands now

The original held-out direction analysis was repeated on all **40 fresh
models**, excluding exact segment joins where the derivative has two-sided
values. Neither one-vector alignment passed the original four-part rule at
any horizon. Their held-out $R^2$ gains remained near zero or negative,
including when analysis was restricted to times with current and future
trajectory error below 1. Thus the negative directional result is not
explained merely by the model's visibly failed runs.

At horizons 0.25, 0.5, and 1.0, an **oracle full-amplification** feature
using $\|\Phi(t+\Delta,t)r(t)\|$ added held-out $R^2$ of +0.066,
+0.267, and +0.369 beyond residual size. With elapsed time and its square
in the baseline and **whole-window bootstrap** uncertainty, its gains
were +0.056 [0.031, 0.083], +0.258 [0.206, 0.395], and +0.442
[0.267, 0.672]. The full tangent propagator $\Phi$ was calculated along
the known reference trajectory, so this is an **explanation-oriented
oracle**, not a deployable prediction method.

The defensible present conclusion is narrow: **for these Lorenz PINNs,
pointwise alignment with one expanding direction is not a reliable
held-out predictor of total future error.** The same pattern held on
fresh initial conditions with the frozen sequential model. More complete
dynamical amplification appears relevant. This does not establish why the
one-vector predictor fails, nor does it validate a practical predictor.
The sequential model was designed after the original test, so its new
validation strengthens a post-hoc follow-up rather than retroactively
turning it into the original preregistered experiment.

## 5. When can a conclusion be presented?

**For tomorrow's presentation:** The professor can be shown a completed
project conclusion about the *original* pointwise hypothesis and a fresh
40-run validation of the improved model. The honest headline is that the
simple directional predictor was unsupported in both studies, while the
sequential model achieved two-unit accuracy in 30/40 new runs and still
failed on certain initial conditions. The oracle finding gives a plausible
next direction. The [validation run log](validation_progress_2026-11-22.md)
and fresh-set figure support the numbers.

**For a positive practical predictor claim:** The project must implement a
causal statistic from past PINN residuals, compare it with residual size,
elapsed time, and simple trailing-history controls, and validate it on those
untouched windows. Its input cannot include the true trajectory or oracle
tangent propagator. The piecewise Duhamel quadrature also needs correction
at segment joins before it is used as mechanistic evidence.

The independent round is complete, so a bounded Lorenz-specific negative
conclusion is available **now**. A wider paper claim would still require
another dynamical system or a validated reference-free predictor. The
result should not be described as proof that residual direction never
matters in chaotic systems.

## Suggested answer to a direct question from the professor

> **“Did the initial hypothesis fail?”** The planned test did not support
> the hypothesis that one pointwise expanding-direction alignment predicts
> future Lorenz PINN error beyond residual magnitude. The frozen improved
> PINN repeated that pattern on new initial conditions. The result is
> specific to this tested predictor and
> setting; the full-vector oracle result suggests that dynamical
> amplification still matters.

> **“What is the contribution now?”** The project has a reproducible
> negative test, a sequential PINN that extends accurate Lorenz trajectories
> to two units in 30/40 new runs, and evidence separating a weak one-vector
> statistic from a stronger full-propagator oracle. The next contribution
> under test is a causal, reference-free residual-history indicator.
