# Directional residuals as error indicators for PINN solutions of chaotic ODEs

Research code for an investigation. Nothing here claims that the approach works,
or that it is new; the experiments were built so that the hypothesis could fail.

**Outcome:** under a decision rule fixed before the main runs, the hypothesis is
**not supported** for standard PINNs on Lorenz-63. The residual's alignment with
a locally expanding direction (either the backward Lyapunov vector $b_1$ or the
finite-time singular vector $s_1$) did not predict future error better than the
residual's size on held-out trajectories. The directional training loss
(Phase 9) was therefore not attempted. See [docs/results.md](docs/results.md).
For the conclusion in relation to the submitted report and reference paper, see
[docs/conclusion_2026-11-21.md](docs/conclusion_2026-11-21.md).
The exploratory interpolation follow-up is documented in
[docs/interpolation_study.md](docs/interpolation_study.md) and its running record
in [docs/interpolation_progress.md](docs/interpolation_progress.md).
The requirements for a paper-ready report are in
[docs/paper_roadmap.md](docs/paper_roadmap.md).
For a short presentation of the results to date, use
[docs/interim_research_update.pdf](docs/interim_research_update.pdf)
([source text](docs/interim_research_update.md)).
In 40 rebuilt runs, a trailing residual average improved held-out prediction at
short horizons, and linear interpolation added a smaller gain over sample-and-hold;
this exploratory result does not change the original directional-alignment verdict.
The current model-improvement work uses consecutive hard-initial-condition PINNs
to extend trajectory accuracy to length 2. The development set gave 35/40
runs below error 1 throughout; a frozen check on new initial conditions gave
**30/40**. The original one-vector directional finding remained negative
on the new runs, while an oracle full-propagator feature added held-out
information at longer horizons. See the
[independent validation log](docs/validation_progress_2026-11-22.md),
[frozen plan](docs/independent_validation_plan.md), and
[chronological model record](docs/model_improvement_progress.md). The
[earlier research update](docs/piecewise_pinn_research_update.pdf) describes
the development set; the professor briefing below contains the latest result.
For a short third-person explanation of the original goal, failed hypothesis,
current result, and next conclusion milestone, use the
[professor briefing](docs/professor_project_brief_2026-11-22.pdf)
([text](docs/professor_project_brief_2026-11-22.md)).
For the 23 November presentation, use the
[four-slide PDF](docs/presentation_2026-11-23.pdf) and
[speaking notes](docs/presentation_talking_points_2026-11-23.md).

## Research question

> Does the component of a PINN's differential-equation residual along locally
> unstable dynamical directions predict future trajectory error better than the
> ordinary residual magnitude?

Two residual vectors can have the same length. If one points along a direction in
which the dynamics stretches perturbations, the error it causes may grow quickly;
if the other points along a contracting direction, its error may shrink. Standard
PINN training and diagnostics use only the length. This project tests whether the
direction carries additional predictive information, using Lorenz-63 as the main
test system and a linear saddle as a sanity check.

## Status

| Phase | Content | Status |
|---|---|---|
| 0 | Linear saddle: equal-size perturbations and residuals ([notes](docs/phase0_linear_saddle.md)) | done |
| 1 | Lorenz-63 right-hand side and analytic Jacobian, checked numerically ([notes](docs/phase1_lorenz_jacobian.md)) | done |
| 2 | High-accuracy reference trajectories; tolerance study ([notes](docs/phase2_lorenz_reference.md)) | done |
| 3 | Tangent dynamics; QR Lyapunov exponents; directions $b_1$ and $s_1$ ([notes](docs/phase3_lyapunov.md)) | done |
| 4 | Baseline PINN; pilot for window length and initial-condition weight ([notes](docs/phase4_pinn.md)) | done |
| 5 | Trajectory error $E(t)$, future error $E_\Delta(t)$, linear (Duhamel) error reconstruction | done |
| 6 | Central experiment, following the [analysis plan](docs/analysis_plan.md) ([results](docs/results.md)) | done |
| 7 | 8 initial conditions × 5 seeds; sensitivity sets | done |
| 8 | Negative controls: random, time-shuffled, least-expanding and flow directions | done |
| 9 | Directional loss: only if Phases 6–8 support the hypothesis | not run: not supported |

## Background in plain language

**Physics-informed neural network (PINN).** A neural network $\hat x_\theta(t)$ that
takes time as input and outputs the state. Instead of fitting data, it is trained
so that it satisfies the differential equation: its time derivative (computed by
automatic differentiation) should equal the right-hand side of the ODE, and it
should match the initial condition.

**Residual.** What is left over when an approximate solution is substituted into the
equation. For $\dot x = f(x)$ and an approximate solution $\hat x(t)$,
$r(t) = \dot{\hat x}(t) - f(\hat x(t))$. The exact solution has $r = 0$. A PINN is
trained by minimising the mean of $\lVert r\rVert^2$ over sampled times, so
$\lVert r\rVert$ is the natural built-in measure of how wrong it is.

**Why the residual is not the error.** The error $e = \hat x - x$ obeys, to first
order, $\dot e = J(x(t))\,e + r(t)$, where $J$ is the Jacobian of $f$. Its solution is

$$
e(t) = \Phi(t, 0)\,e(0) + \int_0^t \Phi(t, s)\, r(s)\, ds ,
$$

where the propagator $\Phi(t, s)$ maps a small perturbation at time $s$ to its
evolved form at time $t$. The residual acts as a stream of small kicks, and each
kick is stretched or squeezed by the dynamics according to its direction. See
[docs/phase0_linear_saddle.md](docs/phase0_linear_saddle.md) for a worked example.

**Stable and unstable directions.** Near a trajectory, the dynamics stretches small
perturbations in some directions (unstable, expanding) and squeezes them in others
(stable, contracting). For $\dot x = x,\ \dot y = -y$, a perturbation along $x$ grows
like $e^{t}$ and one along $y$ shrinks like $e^{-t}$.

**Jacobian and tangent dynamics.** The Jacobian $J(x)$ is the matrix of partial
derivatives of $f$; it describes how $f$ changes for a small change of state.
Small perturbations evolve by the linear "tangent" equation $\dot{\delta} = J(x(t))\,\delta$
along the trajectory, whose solution is $\delta(t) = \Phi(t, s)\,\delta(s)$.

**Lyapunov exponents.** Long-run average exponential growth rates of perturbations:
$\lVert\delta(t)\rVert \approx \lVert\delta(0)\rVert e^{\lambda t}$. A system in
$n$ dimensions has $n$ of them. A positive largest exponent means chaos: nearby
trajectories separate exponentially. For Lorenz-63 with the classical parameters
the commonly quoted values are about $+0.91$, $0$ and $-14.57$.

**Lyapunov vectors.** The directions associated with the exponents. Several
non-equivalent definitions exist, and this project names exactly which one it uses:

| name | definition |
|---|---|
| backward (Gram–Schmidt) vectors $b_i(t)$ | orthonormal basis obtained by propagating perturbations forward from the past and re-orthonormalising with QR (Benettin et al. 1980; Shimada & Nagashima 1979). $b_1$ is the direction past perturbations have lined up with. |
| forward vectors $f_i(t)$ | the analogous orthonormal basis built from the future; $f_1$ is the direction that will grow the most. |
| covariant Lyapunov vectors (CLVs) $v_i(t)$ | directions carried into each other by the dynamics, $\Phi(t, s)\,v_i(s) \parallel v_i(t)$; generally not orthogonal (Ginelli et al. 2007; Kuptsov & Parlitz 2012). $v_1 = b_1$. |
| finite-time singular vectors $s_i(t;\Delta t)$ | right singular vectors of $\Phi(t+\Delta t, t)$; $s_1$ is the perturbation that grows the most over exactly $\Delta t$. |

Phase 0 shows with a 2-D example why the distinction matters: in a non-normal
system, "perpendicular to $b_1$" does not mean "non-growing".

## Hypothesis and how it could fail

For a residual $r(t)$ and a unit expanding direction $u_1(t)$:

* Metric A, total residual: $R = \lVert r\rVert$
* Metric B, residual along $u_1$: $R_u = |r\cdot u_1|$
* Metric C, transverse residual: $R_\perp = \lVert r - (r\cdot u_1)\,u_1\rVert$
* alignment: $A = |r\cdot u_1| / (\lVert r\rVert + \epsilon)$

The hypothesis is supported only if directional information predicts future error
after controlling for $R$ (alignment analysis at fixed $R$, regression evaluated on
held-out times or trajectories) and if it beats the negative controls. It is not
supported if, for example, random or time-shuffled directions predict as well as
$u_1$, if $R$ alone explains the error, or if the effect is confined to a single
seed or trajectory.

## Methodology

* **Reference.** SciPy `solve_ivp` (DOP853, rtol $10^{-12}$, float64), checked against a
  stricter DOP853 run and against LSODA. Each initial condition comes from a long
  run on the attractor, so it has a 30-unit past.
* **Directions.** Oracle (post-hoc) directions computed from the reference
  trajectory, labelled as such: the backward (QR) Lyapunov vector $b_1(t)$ and the
  finite-time singular vector $s_1(t;\Delta t)$ of $\Phi(t+\Delta t, t)$, compared as separate
  predictors.
* **PINN.** 4 × 64 tanh network, Adam, loss $L = L_{\text{ODE}} + \lambda_{\text{IC}} L_{\text{IC}}$ on a
  window of length 1 with a time curriculum, float32; one seed per window
  repeated in float64.
* **Statistics** ([analysis plan](docs/analysis_plan.md), fixed before the main runs).
  Pearson and Spearman correlations on logarithms, log–log scatter plots,
  binned analysis at fixed residual size, and the decisive test: regressions
  $\log E_\Delta = a + b\log R$ versus $a + b\log R + cA$, evaluated on held-out initial
  conditions, with cluster-bootstrap intervals over runs.
* **Replication and controls.** 8 initial conditions × 5 seeds, all runs reported;
  random-direction, time-shuffled, least-expanding and flow-direction controls.

## Repository layout

```text
config/default.yaml                 settings for every experiment (one section each)
src/systems/linear_saddle.py        Phase 0 linear systems, exact solutions, direction utilities
src/systems/lorenz63.py             Lorenz-63 right-hand side, analytic Jacobian, fixed points, Jacobian gate
src/solvers/reference_solver.py     solve_ivp wrapper that records solver settings; tolerance study
src/solvers/lorenz_reference.py     initial conditions with a past, and reference trajectories
src/dynamics/jacobian.py            complex-step and central-difference Jacobians, Jacobian checks
src/dynamics/tangent_dynamics.py    step propagators of the tangent equation and their products
src/dynamics/lyapunov.py            QR method (backward vectors, exponents), finite-time singular vectors
src/pinn/model.py                   the network: rescaled time in, rescaled state out
src/pinn/residual.py                residual by automatic differentiation
src/pinn/train.py                   standard PINN loss, Adam, time curriculum, stacked training of many runs
src/pinn/runs.py                    train, evaluate against the reference, and save one run
src/analysis/projections.py         R, R_u, R_perp and alignment
src/analysis/future_error.py        future error, error created within a horizon, linear reconstruction
src/analysis/metrics.py             correlations, fixed-size alignment effect, held-out regression, bootstrap
src/analysis/plots.py               shared figure style
src/utils/                          checks, reproducibility records, seeding
experiments/01_linear_sanity.py     Phase 0: linear saddle
experiments/02_lorenz_jacobian.py   Phase 1: Lorenz-63 and its Jacobian
experiments/03_lorenz_reference.py  Phase 2: reference trajectories and tolerance study
experiments/04_lorenz_lyapunov.py   Phase 3: exponents, b1 and s1
experiments/05_pinn_pilot.py        Phase 4: choice of window length and initial-condition weight
experiments/06_pinn_runs.py         Phases 4-7: all PINN runs (8 windows x 5 seeds; float64 check)
experiments/07_directional_analysis.py  Phases 5-8: the central experiment and its controls
docs/                               one note per phase, the analysis plan, and the results
tests/                              unit tests (pytest)
results/                            generated outputs, one directory per experiment
```

Experiments are numbered in the order they run. `utils/logging.py` from the
original outline became `utils/reproducibility.py`, to avoid confusion with
Python's `logging` module. No notebook was needed.

Each experiment writes `config.yaml` (settings used), `metadata.json` (time,
command, runtime, Python and package versions, platform, git commit and
whether the code had uncommitted changes), `metrics.json` (results and
pass/fail checks) and `figures/`. Bulky arrays (`data/`, `*.npz`, `*.pt`) are
not tracked in git; the scripts regenerate them. Each PINN run directory holds
`config.yaml`, `model.pt`, `history.json`, `trajectory.npz`, `residuals.npz` and
`metrics.json`; the Lyapunov data shared by all runs of a window are in
`results/lorenz_lyapunov/data/`.

## Reproducing

Run everything from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
python -m pytest                   # unit tests
python experiments/01_linear_sanity.py
python experiments/02_lorenz_jacobian.py
python experiments/03_lorenz_reference.py
python experiments/04_lorenz_lyapunov.py        # about 3 minutes
python experiments/05_pinn_pilot.py             # pilot on window 0 (8 trainings in parallel)
python experiments/06_pinn_runs.py --device cuda                     # 40 main runs as one stack on a GPU
python experiments/06_pinn_runs.py --device cuda --set pinn_runs_lambda10
python experiments/06_pinn_runs.py --set float64_check               # 8 float64 runs on the CPU
python experiments/07_directional_analysis.py   # the planned analysis and decision
python experiments/07_directional_analysis.py --runs results/pinn_runs_lambda10 --out results/analysis_lambda10
python experiments/07_directional_analysis.py --runs results/float64_check --out results/analysis_float64
python experiments/08_alignment_effect_controls.py   # exploratory controls (not in the plan)
python experiments/09_residual_interpolation.py        # exploratory interpolation after regenerating run arrays
```

Without a GPU, drop `--device cuda`: the runs of each window are then trained
in parallel CPU processes (about 1.5 hours for 40 runs on 6 cores). On a GTX
1650 the 40 runs take about 15 minutes. Times above are for that machine.

On Kaggle (files under `/kaggle/input` are read-only, so work in `/kaggle/working`):

```text
!git clone <repository-url> /kaggle/working/directional_pinn   # or copy an uploaded dataset there
%cd /kaggle/working/directional_pinn
!pip install -q -r requirements.txt
!python -m pytest -q
!python experiments/01_linear_sanity.py
!python experiments/02_lorenz_jacobian.py
!python experiments/03_lorenz_reference.py
!python experiments/04_lorenz_lyapunov.py
!python experiments/05_pinn_pilot.py
!python experiments/06_pinn_runs.py --device cuda
!python experiments/07_directional_analysis.py
```

## Results

Phase 0 (linear saddle, ε = 10⁻³): two approximate solutions whose residual size is
identical at every time differ in true error by a factor $e^{t}$ (148 at $t = 5$),
depending only on whether the residual points along the unstable or the stable
direction. At fixed residual size, the error after a horizon $\Delta t$ increases
with alignment, but by at most a factor $e^{\Delta t}$ (constant residual) or
$e^{2\Delta t}$ (state perturbation), i.e. only 5 % and 10.5 % at $\Delta t = 0.05$.
In an isotropic control the direction carries no information. A non-normal variant
shows that the choice of "unstable direction" matters. Details:
[docs/phase0_linear_saddle.md](docs/phase0_linear_saddle.md).

Phase 1 (Lorenz-63): the analytic Jacobian agrees with complex-step derivatives to
a relative error of $1.3\times10^{-16}$ and with central differences to
$2.1\times10^{-10}$, at the fixed points and 1000 random states; a copy with one
sign error is rejected (relative error 1.3). The Jacobian's eigenvalues at the
fixed points and the Hopf threshold $\rho_H \approx 24.74$ match closed forms. Details:
[docs/phase1_lorenz_jacobian.md](docs/phase1_lorenz_jacobian.md).

Phases 2–3: references stay within $10^{-8}$ of a stricter solution for at least 5.8
time units; Lyapunov exponents $0.924, -0.025, -14.566$ over 180 time units; $b_1$
and $s_1$ are typically 50°–75° apart. Details:
[docs/phase2_lorenz_reference.md](docs/phase2_lorenz_reference.md),
[docs/phase3_lyapunov.md](docs/phase3_lyapunov.md).

Phase 4: plain PINN training collapses onto the invariant solution $x = y = 0$;
with a time curriculum the standard PINN follows Lorenz trajectories over one
time unit (about one Lyapunov time) but not over two or more. Details:
[docs/phase4_pinn.md](docs/phase4_pinn.md).

Phases 5–8 (40 runs, 8 initial conditions; [docs/results.md](docs/results.md)):

* Held-out gain in $R^2$ from adding alignment to $\log R$: $-0.008$ to $+0.020$ for $b_1$
  and $+0.017$ to $+0.046$ for $s_1$, never both $\ge 0.02$ and significant; time-shuffled
  $s_1$ and the flow direction do about as well. Verdict: not supported, also with
  initial-condition weight 10 and with float64 training.
* Inside single runs, at fixed residual size, alignment with $b_1$ or $s_1$ goes with
  larger future error, but so does alignment with the neutral flow direction.
* Trained PINN residuals already point almost entirely along $s_1$ (median
  alignment 0.87–0.96), so orientation varies too little to carry information.
* Weighting the whole residual by the dynamics, $\lVert\Phi(t+\Delta t, t)\,r(t)\rVert$, does
  add held-out information (+0.06 in $R^2$ at $\Delta t$ = 0.25 and 0.5).

## Limitations

* **Short windows.** The standard PINN could only learn Lorenz windows of about one
  Lyapunov time (Phase 4). Orientation may matter more over longer horizons, but
  accurate PINNs for those could not be obtained, so that case is untested.
* **Oracle directions.** $b_1$ and $s_1$ come from the reference trajectory, which a
  deployed PINN does not have. Since even these did not help, directions
  estimated from the PINN itself were not studied.
* **Pointwise predictors.** As in the original question, the residual is taken at
  a single time. The positive result for $\lVert\Phi r\rVert$ suggests that predictors
  integrating the residual over time are more promising; they were not tested
  as a pre-registered hypothesis.
* **Scope.** One system (Lorenz-63, classical parameters), one architecture
  (4 × 64 tanh), one optimiser (Adam with a time curriculum), 8 initial
  conditions. Small effects (held-out $R^2$ gains below about 0.02) could have been
  missed by design.
* **Deviations from the plan,** all documented: the PINN pilot's rule was not
  followed literally (its literal choice gives the same conclusion), one
  verification threshold of Phase 2 was re-specified, and one analysis
  (`experiments/08`) is exploratory.

## Related work

This project builds on established work, and its question is deliberately narrow.

* **PINNs:** Lagaris, Likas & Fotiadis (1998); Raissi, Perdikaris & Karniadakis (2019).
* **Residual-based error estimation for PINNs:** for example Hillebrecht & Unger
  (2022), a posteriori error bounds for PINN solutions of ODEs, and Mishra & Molinaro
  (2023), generalization error estimates in terms of training residuals.
* **Lyapunov analysis of chaotic systems:** Shimada & Nagashima (1979); Benettin et
  al. (1980); Ginelli et al. (2007); Kuptsov & Parlitz (2012).
* **Lyapunov-aware time weighting in PINNs:** Turinici (2024) weights the PINN loss in
  time using local Lyapunov exponent estimates; causality-based time weighting:
  Wang, Sankaran & Perdikaris (2024).
* **Error growth along unstable directions in forecasting:** singular vectors in
  numerical weather prediction (Buizza & Palmer 1995) and assimilation in the
  unstable subspace (Trevisan & Uboldi 2004; review by Carrassi et al. 2018).

The specific question here is whether the **orientation of a PINN's residual relative
to local expanding directions** predicts its future trajectory error beyond the
residual's magnitude.

### References

References were assembled during Phase 0 and should be checked against the
original sources before they are cited in any write-up.

* Benettin, Galgani, Giorgilli & Strelcyn (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. *Meccanica* 15.
* Buizza & Palmer (1995). The singular-vector structure of the atmospheric global circulation. *J. Atmos. Sci.* 52.
* Carrassi, Bocquet, Bertino & Evensen (2018). Data assimilation in the geosciences: an overview of methods, issues, and perspectives. *WIREs Climate Change*. [arXiv:1709.02798](https://arxiv.org/abs/1709.02798)
* Ginelli, Poggi, Turchi, Chaté, Livi & Politi (2007). Characterizing dynamics with covariant Lyapunov vectors. *Phys. Rev. Lett.* 99.
* Hillebrecht & Unger (2022). Certified machine learning: a posteriori error estimation for physics-informed neural networks. *IJCNN 2022*. doi:10.1109/IJCNN55064.2022.9892569
* Kuptsov & Parlitz (2012). Theory and computation of covariant Lyapunov vectors. *J. Nonlinear Sci.* 22.
* Lagaris, Likas & Fotiadis (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Trans. Neural Netw.* 9.
* Mishra & Molinaro (2023). Estimates on the generalization error of physics-informed neural networks for approximating PDEs. *IMA J. Numer. Anal.*
* Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks. *J. Comput. Phys.* 378.
* Shimada & Nagashima (1979). A numerical approach to ergodic problem of dissipative dynamical systems. *Prog. Theor. Phys.* 61.
* Trevisan & Uboldi (2004). Assimilation of standard and targeted observations within the unstable subspace of the observation–analysis–forecast cycle system. *J. Atmos. Sci.* 61.
* Turinici (2024). Lyapunov weights to convey the meaning of time in physics-informed neural networks. [arXiv:2407.21642](https://arxiv.org/abs/2407.21642) (later version titled "Regime-aware time weighting for physics-informed neural networks").
* Wang, Sankaran & Perdikaris (2024). Respecting causality for training physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.*
