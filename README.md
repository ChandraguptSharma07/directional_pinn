# Directional residuals as error indicators for PINN solutions of chaotic ODEs

Research code for an ongoing investigation. Nothing here claims that the approach
works, or that it is new; the experiments are built so that the hypothesis can
fail.

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
| 4 | Minimal PINN (Adam) with the standard residual loss | next |
| 5 | Trajectory error $E(t)$ and future error $E_\Delta(t)$ | planned |
| 6 | Central experiment: $R$ vs $R_u$ vs $R_\perp$, alignment at fixed $R$, held-out regression | planned |
| 7 | Multiple seeds, initial conditions and trajectories | planned |
| 8 | Negative controls (random, shuffled and stable directions) | planned |
| 9 | Directional loss: only if Phases 6–8 support the hypothesis | conditional |

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

## Methodology (planned)

* **Reference.** SciPy `solve_ivp` (DOP853, float64, tight tolerances), checked by
  re-solving with stricter tolerances. The experiment horizon is limited to the
  window in which the reference is trustworthy; Lorenz is chaotic, so two numerical
  solutions eventually separate.
* **Directions.** Oracle (post-hoc) directions computed from the reference
  trajectory, labelled as such. Two are computed and compared as separate
  predictors: the backward (QR) Lyapunov vector $b_1(t)$ and the finite-time
  singular vector $s_1(t;\Delta t)$ of $\Phi(t+\Delta t, t)$. Directions estimated from
  the PINN's own trajectory are a separate, later question (deployability).
* **PINN.** Small fully connected network (4 × 64, tanh), Adam, loss
  $L = L_{\text{ODE}} + \lambda_{\text{IC}} L_{\text{IC}}$, float32 on GPU; representative runs
  repeated in float64.
* **Statistics.** Pearson and Spearman correlations on log-transformed quantities,
  log–log scatter plots, binned analysis, bootstrap confidence intervals, and
  regressions $\log E_\Delta = a + b\log R$ versus $a + b\log R + cA$ evaluated on
  held-out data, for several horizons $\Delta t$.
* **Replication and controls.** At least 5 seeds (all reported, including failures);
  several initial conditions; random-direction, shuffled-direction and
  stable-direction controls.

## Repository layout

```text
config/default.yaml              settings for every experiment (one section each)
src/systems/linear_saddle.py     Phase 0 linear systems, exact solutions, direction utilities
src/systems/lorenz63.py          Lorenz-63 right-hand side, analytic Jacobian, fixed points, Jacobian gate
src/dynamics/jacobian.py         complex-step and central-difference Jacobians, Jacobian checks
src/dynamics/tangent_dynamics.py step propagators of the tangent equation and their products
src/dynamics/lyapunov.py         QR method (backward vectors, exponents), finite-time singular vectors
src/solvers/reference_solver.py  solve_ivp wrapper that records solver settings; tolerance study
src/solvers/lorenz_reference.py  initial conditions with a past, and reference trajectories (Phase 2)
src/analysis/projections.py      R, R_u, R_perp and alignment
src/analysis/plots.py            shared figure style
src/utils/checks.py              pass/fail record of numerical checks
src/utils/reproducibility.py     config loading, CSV/JSON output, environment and git info
src/pinn/                        Phase 4 (empty for now)
experiments/01_linear_sanity.py  Phase 0 experiment
experiments/02_lorenz_jacobian.py  Phase 1 checks
experiments/03_lorenz_reference.py Phase 2 reference trajectories
experiments/04_lorenz_lyapunov.py  Phase 3 exponents and expanding directions
docs/phase0_linear_saddle.md     Phase 0 explanation and results
docs/phase1_lorenz_jacobian.md   Phase 1 explanation and results
docs/phase2_lorenz_reference.md  Phase 2 explanation and results
docs/phase3_lyapunov.md          Phase 3 explanation and results
tests/                           unit tests (pytest)
results/                         generated outputs, one directory per run
notebooks/                       exploration notebooks (none yet)
```

Modules from the original outline that are not there yet
(`pinn/{model,residual,train}.py`,
`analysis/{future_error,metrics}.py`, `utils/seed.py`) are added in the phase that
needs them. Experiments are numbered in the order they are built, so the PINN
experiments will follow the reference (03) and Lyapunov (04) scripts.
`utils/logging.py` became `utils/reproducibility.py` to avoid confusion with
Python's `logging` module.

Each run writes `config.yaml` (settings used), `metadata.json` (time, command,
runtime, Python and package versions, platform, git commit when available),
`metrics.json` (results and pass/fail checks), `data/*.csv` (the data behind every
figure) and `figures/*.png|pdf`.

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
python experiments/04_lorenz_lyapunov.py
```

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
```

## Results so far

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

## Limitations

* Expanding directions come from the reference trajectory (oracle analysis); they
  are not available to a PINN in practice without further work.
* The error-propagation picture is linear; it is only accurate while the PINN error
  is small, and PINNs on chaotic systems often fail badly.
* Chaos limits the time window over which a trustworthy reference exists.
* Only Lorenz-63 with classical parameters is planned.
* Correlation is not causation: the residual and the local expanding direction both
  depend on where the trajectory is on the attractor, which can confound the
  comparison. Multiple trajectories and controls are meant to address this.
* Results may depend on which direction is used (backward, forward, covariant or
  finite-time vectors) and on whether the target is the total future error or only
  the part created after time $t$.

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
