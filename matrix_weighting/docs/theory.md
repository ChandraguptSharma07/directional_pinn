# Theory: from Turinici's time weights to matrix weights

This note derives the weights used in this study. Every step is spelled out
for a reader who knows basic linear algebra and ODEs. The derivation follows
Turinici's Propositions 1 and 2 (arXiv:2407.21642v2, Section 3), but keeps
the **direction** of the residual instead of only its size.

## 1. Setting

- **ODE:** dx/dt = f(x) with x ∈ ℝⁿ and x(0) = x₀ given.
- **PINN:** a candidate x̂(t) with exactly the right initial state, x̂(0) = x₀.
- **Residual:** r(t) = dx̂/dt − f(x̂(t)), the amount by which x̂ fails the
  equation at time t.
- **Training loss:** a weighted average of r over a time grid. The question
  is how to weight it.

## 2. How a residual becomes an error

Let e = x − x̂ be the error. Subtract the two equations:

    de/dt = f(x) − f(x̂) − r.

For a small error, f(x) − f(x̂) ≈ J(t) e, where J(t) is the Jacobian of f
along the path. This is the linearization used throughout the earlier study.
Then

    de/dt = J(t) e − r(t),   e(0) = 0.

The solution uses the **tangent propagator** Φ(t, s). This matrix maps a small
displacement at time s to the displacement it has become by time t. It
solves ∂Φ(t, s)/∂t = J(t) Φ(t, s) with Φ(s, s) = I. Variation of constants
(Duhamel's formula) gives

    e(t) = − ∫₀ᵗ Φ(t, s) r(s) ds.                                   (1)

**Reading (1).** The residual at time s is a small push −r(s) ds. By time t
the dynamics have turned it into Φ(t, s)(−r(s) ds). The error is the sum of
all these transported pushes. How much a push grows depends on its
**direction**: Φ stretches some directions and squeezes others.

## 3. Turinici's scalar version

Turinici bounds the stretching by a number λ(t), the one-sided growth rate.
This gives ‖Φ(T, s) v‖ ≤ exp(∫ₛᵀ λ) ‖v‖ for **every** direction v. Then

    ‖e(T)‖ ≤ ∫₀ᵀ exp(∫ₛᵀ λ) ‖r(s)‖ ds.

He then asks which time weight ρ(s) in the loss ∫ ρ ‖r‖² gives the best
worst-case final error, if training can only reach a fixed level of the
loss. The answer is ρ(s) ∝ exp(−∫₀ˢ λ), his Eq. (13). The weight is the same
for every direction, because the bound is.

## 4. Keeping the direction: the window target

We measure accuracy over the whole window, not only at T. Take the L² norm in
time of (1):

    ‖e‖_{L²(0,T)} = ‖ ∫₀ᵀ E_s ds ‖_{L²},   E_s(t) = 1{t > s} Φ(t, s) r(s).

**Minkowski's inequality** says the norm of an integral is at most the
integral of the norms. Applied here:

    ‖e‖_{L²(0,T)} ≤ ∫₀ᵀ ‖E_s‖_{L²} ds = ∫₀ᵀ sqrt( r(s)ᵀ G(s) r(s) ) ds,     (2)

where

    G(s) = ∫ₛᵀ Φ(t, s)ᵀ Φ(t, s) dt.                                  (3)

**What G means.** For a unit push v at time s, vᵀ G(s) v is the squared error
it causes, summed over the rest of the window. The eigenvectors of G(s) are
the push directions; the eigenvalues say how harmful each direction is.

- A push along a strongly contracting direction (rate −κ) dies within about
  1/κ time units. Its eigenvalue is about 1/(2κ), which is small.
- A push along an expanding direction (rate +λ) keeps growing. Its
  eigenvalue is about exp(2λ(T − s)) / (2λ), which is large early in the
  window.

**Final-time target.** If only the final error ‖e(T)‖ matters, the triangle
inequality replaces (2) by ∫ ‖Φ(T, s) r(s)‖ ds. Every statement below then
holds with G(s) replaced by C(s) = Φ(T, s)ᵀ Φ(T, s), the Cauchy–Green
tensor. Its eigenvectors are the finite-time singular vectors s₁, s₂, s₃ of
the original study.

## 5. The best matrix weight

Allow the loss to weight each direction differently: ∫ r(s)ᵀ M(s) r(s) ds
with M(s) symmetric and positive definite. We need a **budget**, the analogue
of Turinici's ∫ ρ = 1. We use ∫₀ᵀ tr M(s) ds = 1. With M = ρ I this is
n ∫ ρ, Turinici's budget up to the constant n.

**Problem.** Training reaches ∫ rᵀ M r ≤ c, but we do not control which r it
reaches. The worst case is

    W(M) = max over r with ∫ rᵀMr ≤ c of ∫ sqrt(rᵀ G r) ds.

We want the M with the smallest W(M) under the budget.

**Step 1: the worst residual for a given M.** Write rᵀ G r = ‖G^{1/2} r‖².
Insert M^{-1/2} M^{1/2}:

    ‖G^{1/2} r‖ ≤ ‖G^{1/2} M^{-1/2}‖ ‖M^{1/2} r‖,

where the first factor is the largest singular value. Applying
Cauchy–Schwarz in time,

    ∫ ‖G^{1/2} r‖ ds ≤ sqrt( ∫ a(s)² ds ) sqrt( ∫ rᵀ M r ds ),   a(s) = ‖G^{1/2} M^{-1/2}‖.

Equality is reached by choosing r(s) along the worst direction with size
proportional to a(s). So W(M) = sqrt(c) · sqrt( ∫ a(s)² ds ), and
a(s)² = λ_max(M^{-1/2} G M^{-1/2}).

**Step 2: the best direction weighting at one time.** Fix s and the amount
m = tr M(s) spent there. Since G ⪯ λ_max(M^{-1/2} G M^{-1/2}) · M, taking
traces gives

    λ_max(M^{-1/2} G M^{-1/2}) ≥ tr G / tr M = tr G / m.

Equality holds when M is proportional to G: then M^{-1/2} G M^{-1/2} is a
multiple of I. So at each time the best choice is **M(s) ∝ G(s)**. This is
the step that brings in direction. Weight each direction in proportion to how
much error it causes. The test `test_trace_inequality_behind_the_minimax_weight`
checks the inequality on random matrices.

**Step 3: the best allocation over time.** Now minimize ∫ tr G(s) / m(s) ds
subject to ∫ m = 1. By Cauchy–Schwarz,

    ∫ tr G / m ≥ (∫ sqrt(tr G))²,

with equality at m(s) ∝ sqrt(tr G(s)). Combining the three steps:

    M*(s) ∝ G(s) / sqrt(tr G(s)),      W(M*) = sqrt(c) ∫₀ᵀ sqrt(tr G(s)) ds.     (4)

## 6. The best weight that ignores direction

Repeat the argument with M = ρ I. In Step 1, a(s)² = λ_max(G)/ρ. In Step 3
the budget is n ∫ ρ = 1. Then ρ*(s) ∝ sqrt(λ_max G(s)), and

    W(ρ* I) = sqrt(c) · sqrt(n) ∫₀ᵀ sqrt(λ_max G(s)) ds.

**Ratio of the two guarantees:**

    W(ρ* I) / W(M*) = sqrt(n) ∫ sqrt(λ_max G) / ∫ sqrt(tr G)   ∈ [1, sqrt(n)].

It is 1 when all directions are equally harmful (G ∝ I). It approaches
sqrt(n) when one direction dominates. For Lorenz-63 (n = 3) the cap is 1.73.
On 40 attractor windows, the final-time version gave 1.45 at T = 0.25 and
1.72 at T = 2. So the guaranteed improvement is modest in three dimensions.
It can be larger only in systems with many dimensions, of which few are
harmful.

## 7. Why training might gain more than the guarantee

The guarantee assumes training reaches the same weighted-loss level whatever
the weight. Gradient descent does not behave that way.

Under the plain loss, an early push along an expanding direction is cheap: it
costs one small residual and causes an error that grows like exp(λT). Errors
that grow like this have a small residual compared with their size, so the
plain loss barely sees them. Our global T = 2 Lorenz PINNs show this failure:
median residual 0.05 against median trajectory error 17.

The Gramian weight is the time-diagonal part of the exact linearized window
error, ∫∫ r(s)ᵀ K(s, s′) r(s′) ds ds′. Its diagonal is K(s, s) = G(s). The
weighted loss is therefore closer to the quantity we actually care about,
the error. It can act like a **preconditioner**, a change that makes the loss
landscape better conditioned. Whether it does in practice is what Gates 0–2
test.

## 8. Computing G without the reference

The weights need Φ along the path. The plan computes it two ways:

- **oracle:** along the reference solution;
- **self-consistent:** along the PINN's current path x̂(t), with no reference
  needed. Turinici's Eq. (17) likewise uses the PINN's own state.

On the grid t₀, …, t_{N−1} with step h, the step propagator is
Φᵢ = expm(h (Jᵢ + Jᵢ₊₁)/2). Splitting (3) at t_{i+1}:

    G(tᵢ) = ∫_{tᵢ}^{tᵢ₊₁} Φ(t, tᵢ)ᵀΦ(t, tᵢ) dt + Φᵢᵀ G(tᵢ₊₁) Φᵢ.

The first piece is ≈ (h/2)(I + ΦᵢᵀΦᵢ) by the trapezoidal rule. This gives the
backward recursion in `mw/weights.py`, starting from G(T) = 0. The same
recursion in continuous time is the matrix ODE −dG/dt = JᵀG + GJ + I with
G(T) = 0.

## 9. Assumptions and limits

- The error identity (1) is a linearization. For Lorenz-63 the earlier study
  found about 6% median mismatch.
- The minimax argument bounds a worst case under a budget assumption. It does
  not predict a training run.
- The self-consistent weight uses the PINN's own, possibly wrong, path. That
  is why the oracle version is run alongside it.
- The initial condition is exact (hard constraint), as in Turinici's paper.
