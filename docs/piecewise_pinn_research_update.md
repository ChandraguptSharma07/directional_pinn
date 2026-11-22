# Longer Lorenz PINN trajectories and the residual-direction hypothesis

> **Independent-validation update, 22 November 2026:** This document's
> 35/40 model-accuracy figure is the *development set*. A frozen evaluation
> on new initial conditions gave **30/40** runs below error 1 throughout
> $T=2$, and **32/40** for at least 90% of it. Neither one-vector alignment
> passed the decision rule on the new set. The full reference-based oracle
> signal persisted at longer horizons. For the latest presentation-ready
> conclusion, use the [professor briefing](professor_project_brief_2026-11-22.pdf)
> and [validation log](validation_progress_2026-11-22.md).

*Research update, 22 November 2026. Exploratory follow-up to the submitted project report.*

## Question and main result

The submitted [project report](Project%20Report%20-%20Differential%20Equations%20-2.pdf)
discussed Turinici's [time-weighted PINN paper](2407.21642v2.pdf) and this
project's separate question: does the **direction** of a trained PINN's ODE
residual predict future Lorenz-63 trajectory error beyond residual magnitude?
The original preregistered test on 40 standard PINNs found no support for
pointwise alignment with either the backward Lyapunov vector $b_1$ or the
leading finite-time singular vector $s_1$. Those PINNs were reliable only
over time windows of length 1.

We now have a PINN construction that tracks the reference for **two time
units in 35 of 40 runs** across eight initial-condition windows and five
seeds. Repeating the direction analysis on those more accurate trajectories
still finds no held-out benefit from either one-vector alignment. An oracle
feature using the **full tangent propagator** does add held-out predictive
information at longer horizons. This follow-up was designed after inspecting
the earlier results, so it is evidence for a promising mechanism and model
improvement, not a new confirmatory verdict.

## Model and validation

We call the construction a **sequential time-domain decomposition PINN**,
or a **time-marching PINN**. The literature uses these terms for training
separate networks on successive time intervals. It is closely related to
[hard-constrained sequential PINNs](https://arxiv.org/abs/2403.03223), but
our float32 endpoint transfer leaves state jumps of about $10^{-6}$ and we
do not enforce derivative continuity, so we do not claim to reproduce that
paper's exact-continuity method. The broader time-marching terminology is
also used in [temporal-decomposition work](https://arxiv.org/abs/2302.14227).

Each run trains eight consecutive PINNs on intervals of length 0.25. Every
network has four 64-unit tanh layers and satisfies its starting state exactly:

$$\hat x_k(\tau)=x_{k,0}+\frac{\tau}{0.25}\bigl[g_k(\tau)-x_{k,0}\bigr].$$

The first state is the supplied initial condition. Each later segment starts
from the **previous PINN's predicted endpoint**; no true state from inside
the window is used for training. Each segment receives 4,000 Adam updates
with 256 stratified collocation times per update. Reference states are used
only after training to measure error. The high-accuracy reference and tangent
data passed the project's existing numerical checks.

On the window-0 pilot, the original single-network hard-initial-condition
PINN had median error 6.052 over $T=2$. The piecewise PINN had median error
0.0259 and maximum error 0.154. The former used 20,000 updates at 1,024
collocation points; the latter used 32,000 updates total at 256 points per
update. This is a comparison of complete training setups, not an isolated
estimate of the effect of segmentation.

The settings were then held fixed for windows 1–8 and seeds 0–4:

| Model-quality measure | Result |
|---|---:|
| Runs | 40 |
| Median of run-level median errors | 0.0433 |
| Median of run-level maximum errors | 0.1823 |
| Runs below error 1 throughout $T=2$ | 35/40 |
| Runs below error 1 for at least 90% of $T=2$ | 35/40 |

All five failures are on window 1, where error rises late; their maximum
errors range from about 3.05 to 8.21. Every run on windows 2–8 stays below
error 1 throughout. The two figures show the pilot trajectory and the full
validation distribution:

- [Pilot trajectories and errors](figures/piecewise_pinn_t2_pilot.pdf)
- [Forty-run accuracy distribution](figures/piecewise_pinn_validation_40runs.pdf)

## Revisit of the original prediction question

The original analysis code was applied to all 40 piecewise runs. Exact
segment joins were excluded from predictor samples because the derivative
has different left and right values there. At each horizon, prediction was
evaluated on an entirely held-out initial-condition window. The baseline
used log residual norm; each directional model added one alignment. Neither
$b_1$ nor $s_1$ passed any horizon of the original four-part rule.

| Future horizon | Gain in held-out $R^2$ from $b_1$ alignment | Gain from $s_1$ alignment | Gain from oracle $\|\Phi r\|$ |
|---:|---:|---:|---:|
| 0.05 | −0.0003 | −0.0041 | −0.0005 |
| 0.10 | −0.0001 | −0.0020 | +0.0027 |
| 0.25 | −0.0099 | +0.0029 | +0.0513 |
| 0.50 | −0.0176 | +0.0010 | +0.1948 |
| 1.00 | −0.0002 | −0.0013 | +0.3068 |

The oracle gain persists after adding elapsed time and its square to the
residual-size baseline. At horizons 0.25, 0.5, and 1.0, its gains are
respectively **+0.046, +0.191, and +0.360**. Even after removing the
late-failing window 1, they are **+0.026, +0.135, and +0.169**. Adding
current true error to the time-controlled baseline also leaves positive
oracle gains. Thus the longer-horizon oracle association is not explained
only by elapsed time or window 1 in these diagnostics. The oracle requires
the reference trajectory; it is **not** a deployable predictor.

## Interpretation and limits

The construction makes a longer and usually accurate Lorenz PINN trajectory
available for studying the original question. The negative one-vector
result persists, while full dynamical amplification is associated with
future error. This is consistent with the error equation, where residuals
from earlier times are propagated and accumulated. It does not prove that
one particular training mechanism causes the negative result, and it does
not make a one-time oracle feature a substitute for integrating residual
history.

This analysis is post-hoc relative to the original preregistered experiment.
There are eight distinct initial-condition windows, with five seeds sharing
each window; 40 runs are not 40 independent trajectories. The piecewise
state is continuous to about $10^{-6}$ at joins, but derivative jumps reach
4 in the pilot. Exact joins were removed from predictor samples. The older
smooth-trajectory Duhamel reconstruction diagnostic uses uncorrected
quadrature at those joins and should not be interpreted for this model.
The model fails late on all seeds of window 1, so reliability across longer
horizons and new initial conditions remains open.

## Next decisive work

Freeze this model and the time-controlled analysis before testing **new
initial-condition windows** not used in model development. Use a
window-level uncertainty analysis. For a practical residual-based predictor,
test a causal integral of past residuals against pointwise magnitude, time,
and simple trailing-history controls, without giving the predictor the true
reference trajectory. For the piecewise model, use a two-sided quadrature
when checking Duhamel reconstruction across segment joins.

## Reproducibility

The detailed chronological record, including unsuccessful hard-IC,
early-weighting, and L-BFGS trials, is in
[model_improvement_progress.md](model_improvement_progress.md). The fixed
model is implemented in `experiments/13_piecewise_pinn_pilot.py`, the batch
in `experiments/17_piecewise_batch.py`, the direction analysis in
`experiments/07_directional_analysis.py`, and the oracle controls in
`experiments/19_piecewise_oracle_controls.py`. Numerical outputs are in
`results/piecewise_pinn_pilot/`,
`results/piecewise_40runs_directional_analysis_no_joins/`, and
`results/piecewise_40runs_oracle_controls/`.
