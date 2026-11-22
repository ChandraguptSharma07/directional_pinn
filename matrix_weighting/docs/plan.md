# Plan: direction-aware (matrix) residual weighting for PINNs

*Written on 22 November 2026, before any PINN of this study was trained. It
is a new study with its own hypotheses. It does not change the verdicts of
the original Lorenz study or of the dial study.*

## 1. Why this study

Turinici's *Regime-Aware Time Weighting for Physics-Informed Neural Networks*
(arXiv:2407.21642v2) weights the PINN residual in **time**. A residual at time
t is a small push on the solution, and the dynamics amplify that push by the
time T. Turinici bounds the amplification by one number per time,
exp(∫ λ), and chooses time weights ρ(t) ∝ exp(−∫_0^t λ) that minimize the
worst-case final error. One number per time means **the same weight for every
direction** of the residual.

The earlier work in this repository found:

- a residual's alignment with **one** expanding direction did not predict later
  error (original study and fresh-window validation);
- the **full** tangent propagator applied to the residual, ‖Φ r‖, did add
  held-out information about later error (oracle analysis);
- the ported Turinici weighting did not improve our PINNs.

This study moves the full propagator from diagnosis into the **training loss**.

## 2. Theory in brief

Write the PINN as x̂(t) with exact initial state and residual
r(t) = dx̂/dt − f(x̂). To first order the error e = x − x̂ satisfies
de/dt = J(t) e − r, e(0) = 0, so

    e(t) = − ∫_0^t Φ(t, s) r(s) ds,

where Φ(t, s) is the tangent propagator (Phase 3 of the original study).

**Window target.** By Minkowski's inequality,

    ‖e‖_{L²(0,T)} ≤ ∫_0^T sqrt( r(s)ᵀ G(s) r(s) ) ds,
    G(s) = ∫_s^T Φ(τ, s)ᵀ Φ(τ, s) dτ.

G(s) is the error, summed over the rest of the window, that a unit push at
time s creates in each direction. Its eigenvectors are the directions of the
push; its eigenvalues say how much error each direction causes.

**Minimax weight.** Follow Turinici's Proposition 2, but allow a matrix weight
M(s) ⪰ 0 in the loss ∫ r(s)ᵀ M(s) r(s) ds, with the budget ∫ tr M = 1 in
place of his ∫ ρ = 1. The weight that minimizes the worst-case bound on the
window error is

    M*(s) ∝ G(s) / sqrt(tr G(s)).

The best weight that ignores direction (M = ρ I) is ρ*(s) ∝ sqrt(λ_max G(s)).
The ratio of the two guaranteed bounds lies between 1 and sqrt(n), where n is
the state dimension. Turinici's rule is the direction-blind case built on a
looser scalar bound. Replacing G(s) by Φ(T, s)ᵀΦ(T, s) gives the analogous
result for the final-time error. Proofs are in `docs/theory.md`.

**What the theory does and does not promise.** It promises a better worst-case
*bound*, at most sqrt(n) better. For Lorenz-63 that is at most 1.73, and a
scratch computation on 40 attractor windows gave 1.45 to 1.72 for windows of
length 0.25 to 2. Any larger gain must come from training itself: the plain
loss pays full price for pushes along strongly contracting directions, which
cause little error, and little for early pushes along expanding directions,
which cause much. Whether gradient descent turns the reweighting into better
trajectories is the empirical question of this study.

## 3. Hypotheses

- **H1 (mechanism, known directions).** With directions computed along the
  reference solution (oracle), the matrix weight M beats the best
  direction-blind weight S at matched compute, and the gain grows with the
  stretching gap.
- **H2 (usable method).** With weights computed along the PINN's own current
  trajectory, the matrix weight beats the direction-blind version, uniform
  weighting, and Turinici's weighting on Lorenz-63 windows.
- **H3 (dimension).** The gain of the matrix weight over uniform weighting
  grows with the number of strongly contracting directions.

## 4. Training weights compared

All weights are held fixed within an optimizer step (no gradient flows through
them). Every weight is rescaled so that the average of tr W over the grid
equals n. Uniform weighting has W = I and needs no rescaling.

| code | weight W(t_i) | source of directions |
|---|---|---|
| `U` | I | none |
| `TUR` | ρ_i I, ρ_i ∝ exp(−h Σ_{j≤i} λ_j), λ = ⟨f(x̂) − f(0), x̂⟩ / (‖x̂‖² + 1e-10) | PINN state, every step (Turinici Eqs. 13 and 17, as ported in `src/pinn/train.py`) |
| `S-orc` | sqrt(λ_max G_i) I | reference trajectory (oracle) |
| `M-orc` | G_i / sqrt(tr G_i) | reference trajectory (oracle) |
| `S-self` | sqrt(λ_max G_i) I | current PINN trajectory, every 10 steps |
| `M-self` | G_i / sqrt(tr G_i) | current PINN trajectory, every 10 steps |

For a linear system J is constant, so the self and oracle versions are the
same weight. Only `U`, `TUR`, `S` and `M` are run there.

**Computing G on the grid.** Use the Jacobians J_i at the grid states. The
step propagator is Φ_i = expm(h (J_i + J_{i+1}) / 2). Then run the recursion
G_i = Φ_iᵀ G_{i+1} Φ_i + (h/2)(I + Φ_iᵀ Φ_i), starting from G_{N−1} = 0. This
is the trapezoidal rule for the integral defining G. It is computed in
float64.

## 5. Common training setup

These settings are fixed for all methods and gates unless a gate says
otherwise.

- Each run trains a batch of independent networks, one per initial condition
  and seed. Every network is a 4 × 64 tanh MLP, Xavier-normal initialized with
  seed 1000 · seed + initial-condition id. A given (initial condition, seed)
  therefore starts from the same network under every method.
- The input is τ = 2t/T − 1. The raw output is mean + scale · net(τ). The
  initial condition is imposed exactly: x̂(t) = x0 + (t/T)(raw − x0).
- The residual r = dx̂/dt − f(x̂) uses the exact time derivative, propagated
  forward through the layers.
- Collocation is a fixed uniform grid of N = 512 times on [0, T], endpoints
  included.
- The loss is (1/N) Σ_i r_iᵀ W_i r_i.
- The optimizer is Adam, 20,000 steps, with the learning rate decaying
  exponentially from 1e-3 to 1e-5. Training is in float32.
- Evaluation uses 2,001 equally spaced times against a reference computed with
  DOP853 (rtol 1e-12, atol 1e-12), or the exact formula for the linear saddle.
- Output scaling (mean, scale):
  - Lorenz systems: (0, 0, 25) and 10 for every component, as in `src/pinn/train.py`. Each damped mode q_j uses mean 0 and scale 10.
  - Dial systems: per system, the midrange and half-range over the reference windows 0–8 on [0, T].

## 6. Outcomes

- **Primary:** E_rms = sqrt(mean over the evaluation grid of ‖x̂ − x‖²).
  Comparisons use log10 E_rms. Non-finite errors and errors above 1e3 are set
  to 1e3.
- **Secondary:**
  - maximum error E_max and final error E_T;
  - Lorenz success, E_max < 1 over the whole window;
  - residual RMS.

**Paired contrast.** For two methods A and B, the contrast is
Δ = log10 E_rms(A) − log10 E_rms(B) for the same initial condition and seed.
It is averaged over seeds within each initial condition. The study then
reports the mean of these per-condition values, with a 95% percentile
bootstrap interval: 10,000 resamples of initial conditions, seed 0.

**"A beats B"** means both:

- mean Δ ≤ −0.176, that is, A has at least 1.5 times lower error;
- the interval's upper end is below 0.

**"A is worse than B"** means the interval's lower end is above 0.

## 7. Gate 0: dial systems (nixxa)

- **Systems:** the dial study's stiff linear saddle with κ ∈ {1, 4, 16, 64}
  (angle π/6), and its Davis–Skodje system with γ ∈ {3, 6, 18, 66}.
- **Window:** T = 2.
- **Initial conditions:** dial windows 1–8 (window 0 is used only for scaling and the timing pilot).
- **Seeds:** 0, 1, 2.
- **Methods:** saddle: `U`, `TUR`, `S`, `M`. Davis–Skodje: all six.

**Decision for H1:**

- **Supported:** in at least one family, `M` beats `S` (oracle versions) at the
  two largest dial settings. In addition, its mean gain −Δ at the largest
  setting exceeds its gain at the smallest.
- **Rejected:** `M` beats `S` at no setting of either family.
- **Partial:** any other outcome.

Secondary contrasts (all reported):

- `M` against `U` and `TUR`;
- `S` against `U`, to separate the effect of time profile from direction;
- self against oracle on Davis–Skodje.

## 8. Gate 1: Lorenz-63 (nixxa)

- **Windows:** single windows with T ∈ {0.5, 1, 2}.
- **Initial conditions:** 12 fresh attractor states. Spin up from (1, 1, 1)
  for 100 time units, then sample 12 times uniformly in [0, 600] along one
  trajectory with NumPy seed 2026112201, sorted. The states are frozen in
  `results/gate1/initial_conditions.json` before training.
- **Seeds:** 0, 1, 2.
- **Methods:** all six.

**Decision for H2.**

- **Supported:** `M-self` beats each of `S-self`, `U` and `TUR` at one or more
  horizons, and is worse than none of them at any horizon.
- **Rejected:** `M-self` beats none of the three at any horizon.
- **Partial:** any other outcome.

`M-orc` against `S-orc` at each horizon is the mechanism check. Success counts
(E_max < 1) are reported for every method and horizon.

## 9. Gate 2: dimension dial (Kaggle)

- **System:** Lorenz-63 drives k damped modes, dq_j/dt = γ_j (x − q_j), with
  γ_j log-spaced in [5, 50] and k ∈ {0, 3, 9, 29}. The state dimension is
  n = 3 + k. The modes never feed back into the Lorenz part.
- **Window:** T = 1.
- **Initial conditions:** the 12 states of Gate 1 for (x, y, z), with q_j(0) = x(0).
- **Seeds:** 0, 1, 2.
- **Methods:** `U`, `TUR`, `S-self`, `M-self`, `S-orc`, `M-orc`.
- **Primary outcome for H3:** E_rms of the Lorenz components (x, y, z).

**Decision for H3.** For each initial condition, compute the gain of `M-self`
over `U` at k = 29 minus the same gain at k = 0. H3 is supported if the
bootstrap interval of the mean difference lies above 0.

## 10. Side runs (Kaggle, exploratory, labelled as such)

- Final-time weight: `M-final` uses Φ(T, s)ᵀΦ(T, s) in place of G(s).
- Causal weighting (Wang, Sankaran and Perdikaris) with ε ∈ {0.1, 1, 10}:
  ρ_i ∝ exp(−ε h Σ_{j<i} ‖r_j‖²).
- Budget and architecture sensitivity on Lorenz T = 1: 50,000 steps, width
  128, and N = 1,024.
- Turinici's own benchmark: initial state (1, 1, 1), T = 0.5, his 5 × 20 tanh
  network, learning rate 0.01, 2,000 steps, 256 grid times.

Side runs can suggest follow-ups. They cannot change a gate decision.

## 11. Reporting rules

- Every run is reported. Failed or non-finite runs count with the capped
  error.
- Oracle weights are always labelled oracle.
- No novelty or superiority claim beyond the decision rules.
- A short literature check must come before any novelty statement. The
  closest known work uses dual-weighted-residual error estimates to choose
  PINN collocation points (arXiv 2604.01835).
- Any change to this plan after training starts is recorded as a dated
  amendment, with its reason.

## Amendment 1 (22 November 2026, evening): Gate 3, more systems

*Written after the first side results (GPU replicate of Gate 1, part of
Gate 2) but before any network of Gate 3 was trained. Gates 0–2 and their
rules are unchanged.*

**Why.** The question includes whether any gain holds for a *group* of
systems, not only Lorenz-63. Gate 3 tests the same claim on two further
chaotic systems, chosen before any of their runs:

- **Rössler:** a = b = 0.2, c = 5.7. One weakly expanding direction (about
  0.07), one neutral, one strongly contracting (about −5.4).
- **Lorenz-96:** N = 8, F = 8. Several expanding directions.

**Setup.**

- Window lengths: Rössler T ∈ {5, 10}; Lorenz-96 T ∈ {1, 2}.
- Initial conditions: 12 attractor states per system, frozen in
  `results/gate3/initial_conditions_<system>.json` before training. Spin-up
  100 from a fixed start, then 12 sorted uniform times in [0, 600], NumPy seed
  2026112202.
- Seeds 0, 1, 2. Training settings as in section 5.
- Output scaling: midrange and half-range over a 600-unit reference
  trajectory.
- Platform: Kaggle GPU.

**Methods.** `U`, `TUR`, `S-self`, `M-self`, plus one new exploratory weight:

- `D-self`, direction only: W_i ∝ n G_i / tr G_i. Every time gets the same
  total weight, so time emphasis is separated from direction.

`D-orc` and `D-self` are also run on Lorenz T = 1 as a side run.

**Decision (H4).** The rule is the H2 rule of section 8, applied per system:

- **Supported** for a system: `M-self` beats each of `S-self`, `U` and `TUR`
  at one or more of its window lengths, and is worse than none at any.
- **Rejected:** `M-self` beats none of the three at any length.
- **Partial:** any other outcome.

`D-self` results are reported descriptively only.
