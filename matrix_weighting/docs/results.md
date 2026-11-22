# Results: direction-aware (matrix) residual weighting for PINNs

*Draft of 22 November 2026, night. The Gate 1 section is completed once the
nixxa runs finish. Every number comes from `results/` and the planned scripts.
Contrast intervals resample whole initial conditions. "Success" means the
trajectory error stays below 1 over the whole window. Side runs are marked
exploratory.*

## 1. Summary in plain words

**The method.** We extended Turinici's time weighting of the PINN residual
(arXiv:2407.21642v2) from one number per time to one matrix per time. The
matrix comes from the tangent propagator of the ODE:

    W(t) ∝ G(t) / sqrt(tr G(t)),   G(t) = ∫_t^T Φ(s, t)ᵀ Φ(s, t) ds.

G(t) says how much error a residual at time t causes in each direction over
the rest of the window. The same worst-case argument as Turinici's gives this
as the best weight for the window error (`docs/theory.md`). His weight is the
special case that ignores direction. The weight can be computed along the
PINN's own current path, with no reference solution (`M-self`).

**What we found.**

1. **Weights from the tangent Gramian rescue training where the plain and
   Turinici losses fail.** On single Lorenz-63 windows of length 1, uniform
   weighting succeeds in 7 of 36 runs and Turinici's weighting in 5 of 36. The
   Gramian weights computed along the PINN's own path succeed in 24 (scalar)
   to 27 (matrix) of 36. The error is typically about 100 times lower.
   - **Robustness:** the uniform and Turinici failures persist with 2.5 times
     more steps, a wider network, or a denser grid; the Gramian-weight
     success persists too.
2. **Direction matters most when many directions are harmless.**
   - **Plain Lorenz-63, self-consistent weights:** the matrix weight is not
     reliably better than the direction-blind scalar version (×0.52, interval
     includes 1).
   - **Oracle (fixed) weights:** the matrix version succeeds in 23 of 36
     runs, the scalar version in 9 of 36.
   - **Damped modes added** (Gate 2), with gains relative to the scalar
     version:

     | modes k | `M-self` gain |
     |---|---|
     | 3 | ×0.13 |
     | 9 | ×0.02 |
     | 29 | ×0.01 |

   - **Rössler, T = 10:** ×0.04.
   - **Lorenz-96, T = 1:** ×0.41.
3. **On easy problems the matrix weight costs a little accuracy.** On the
   two-dimensional dial systems, every method succeeds, and there the matrix
   weight gives 1.2 to 4 times larger errors at small and medium stiffness. It
   helps only at the stiffest saddle (κ = 64).
4. **The target matters: the window, not the final time.** The final-time
   matrix weight (Turinici's target) fails at T ≥ 1, with 0 or 1 successes
   in 36.
5. **Time-marching remains the simplest route to long horizons.** On Lorenz
   to horizon 4:
   - Uniform weighting with segments of 0.5 succeeds in 27 of 36 runs.
   - The matrix weight makes segments of length 1 trainable: 15 of 36,
     against 0 of 36 for uniform.
   - Those longer segments still do not beat many short ones.
6. **Mechanism.** Matrix-weighted networks put 60–73% of their leftover
   residual along the direction the dynamics erase fastest. Uniformly
   weighted networks put about 60% along the most harmful direction.

**Planned verdicts.**

| hypothesis | where | verdict |
|---|---|---|
| H1: mechanism, known directions | Gate 0, dial systems | **partial** |
| H2: usable method | Gate 1, Lorenz-63 on nixxa | **partial**: beats `U` and `TUR` at T = 1 and 2; never beats `S-self` |
| H3: gain over uniform grows with dimension | Gate 2 | **not supported** (the gain is already ≈ ×0.01 at k = 0) |
| H4: same rule on more systems | Gate 3 | **supported** for both Rössler and Lorenz-96 |

## 2. Setup (fixed in `docs/plan.md` before training)

**Network and training.**

- A 4 × 64 tanh MLP. The initial condition is exact by default; the soft form
  was used only in the side check of section 5.
- Collocation is a fixed uniform grid of 512 times.
- Adam runs 20,000 steps, with the learning rate decaying from 1e-3 to 1e-5.
  Training is in float32.
- Every method uses the same networks, initializations, grid, steps and
  learning rates; only the weights differ.

**Weights compared.**

| code | weight |
|---|---|
| `U` | uniform |
| `TUR` | Turinici's Eqs. (13) and (17) |
| `S` | best direction-blind weight from the Gramian, sqrt(λ_max G) I |
| `M` | matrix weight, G / sqrt(tr G) |

Suffix `-orc`: G computed along the reference (oracle). Suffix `-self`: G
computed along the PINN's own path, refreshed every 10 steps.

**Scale.** 12 attractor initial conditions (8 for the dial systems), times 3
seeds, gives 36 (or 24) networks per method and setting.

**Statistics.** Paired contrasts of log10 E_rms, averaged over seeds within
each initial condition. 95% intervals come from 10,000 bootstrap resamples of
initial conditions. "A beats B" needs at least 1.5 times lower error with the
interval below 0.

## 3. Results by gate

### Gate 0: two-dimensional dial systems (H1 partial)

Paired contrasts, `M-orc` against `S-orc` (fold change in E_rms):

| system | dial 2 | dial 5 | dial 17 | dial 65 |
|---|---|---|---|---|
| linear saddle | ×1.91, worse | ×3.06, worse | ×4.28, worse | **×0.50, beats** |
| Davis–Skodje | ×1.23, worse | ×2.46, worse | ×1.78, worse | ×1.03, no decision |

All methods succeed here (24/24) except `U` and `TUR` at κ = 64 (20/24 and
21/24). At κ = 64 the matrix weight also beats `U` (×0.17) and `TUR` (×0.30).
Figure: `results/figures/gate0_errors.png`.

### Gate 1: Lorenz-63 single windows (H2)

*To be completed from the nixxa runs.*

Preview from the GPU replicate (side), with medians of log10 E_rms:

| T | `U` | `TUR` | `S-orc` | `M-orc` | `S-self` | `M-self` |
|---|---|---|---|---|---|---|
| 0.5 | 36/36, −3.05 | 36/36, −3.01 | 36/36, −3.02 | 36/36, −2.93 | 36/36, −2.99 | 36/36, −2.88 |
| 1 | 7/36, 1.29 | 5/36, 1.29 | 9/36, 1.27 | 23/36, −2.36 | 24/36, −2.27 | 27/36, −2.29 |
| 2 | 0/36, 1.40 | 0/36, 1.38 | 6/36, 1.35 | 3/36, 1.31 | 5/36, 1.32 | 8/36, 1.15 |

### Gate 2: Lorenz-63 driving k damped modes (H3 not supported)

Success counts on the Lorenz part:

| k (n = 3 + k) | `U` | `TUR` | `S-orc` | `M-orc` | `S-self` | `M-self` |
|---|---|---|---|---|---|---|
| 0 | 7 | 5 | 9 | 23 | 24 | **27** |
| 3 | 7 | 4 | 5 | 17 | 20 | **27** |
| 9 | 3 | 0 | 4 | 10 | 7 | **27** |
| 29 | 0 | 0 | 3 | 5 | 5 | **27** |

`M-self` against `S-self`:

| k | fold change | verdict |
|---|---|---|
| 0 | ×0.52 | no decision |
| 3 | ×0.13 | beats |
| 9 | ×0.02 | beats |
| 29 | ×0.01 | beats |

The planned H3 contrast is against `U`. It is already about ×0.01 at k = 0
and does not grow, so H3 is not supported. Figure:
`results/figures/gate2_errors.png`.

### Gate 3: Rössler and Lorenz-96 (H4 supported for both)

| system, T | `M-self` vs `S-self` | vs `U` | vs `TUR` | success: `M-self` / `S-self` / `U` / `TUR` |
|---|---|---|---|---|
| Rössler, 5 | ×0.36, no decision | ×0.17, beats | ×0.16, beats | 25 / 21 / 18 / 19 |
| Rössler, 10 | ×0.04, beats | ×0.03, beats | ×0.02, beats | 30 / 13 / 4 / 1 |
| Lorenz-96, 1 | ×0.41, beats | ×0.07, beats | ×0.02, beats | 36 / 36 / 29 / 13 |
| Lorenz-96, 2 | ×0.42, no decision | ×0.19, beats | ×0.16, beats | 10 / 1 / 0 / 0 |

## 4. Side runs (exploratory)

**Sensitivity, Lorenz T = 1** (median log10 E_rms):

| variant | `U` | `TUR` | `S-self` | `M-self` |
|---|---|---|---|---|
| 50,000 steps | 1.29 | 1.29 | −2.96 | −2.65 |
| width 128 | 1.29 | 1.29 | −2.47 | −2.45 |
| grid 1,024 | 1.29 | 1.29 | −2.67 | −2.25 |

**Final-time matrix weight** (Φ(T, t)ᵀΦ(T, t) in place of G): successes 21
and 20 of 36 at T = 0.5, then 1 and 0 at T = 1, and 0 at T = 2.

**Direction only** (same total weight at every time), Lorenz T = 1:

- `D-self`: 23/36, median −1.56.
- `D-orc`: 10/36, median 1.21.

**Time-marching to horizon 4**, successes:

| segment length | `U` | `S-self` | `M-self` |
|---|---|---|---|
| 0.5 | 27/36 | 24/36 | 25/36 |
| 1.0 | 0/36 | 4/36 | 15/36 |

**Causal weighting, Turinici's own benchmark, soft initial condition:** to be
added when those runs finish.

**Mechanism check** (`results/*/mechanism.json`), residual energy along the
least harmful direction of G:

- Lorenz T = 1: 60–73% for the matrix weights, against 1–4% for `U`, `TUR`
  and `S-orc`.
- Lorenz T = 0.5, where all methods succeed: 37–43% for the matrix weights,
  against 17–22% for `U` and `TUR`.

## 5. Limitations

- **The theory is a linearized worst-case bound.** It does not predict a
  training run.
- **The self-consistent weights mix two effects:**
  - direction;
  - an automatic causal time schedule in early training (section 3 of
    `docs/progress.md`).
- **All results are for small MLPs on ODEs** at one optimizer setting. PINNs
  are not competitive with classical integrators for forward ODE problems.
  The value here is for PINN training methodology, and for settings where a
  long single window is needed.
- **Cost.** The self-consistent weights cost about 1.5 to 2.5 times the GPU
  time of uniform weighting at the same step count. The extra time is
  Gramian updates every 10 steps.
- **Hardware.** Gates 0, 2 and 3 ran on Kaggle T4 GPUs; Gate 1 ran on a CPU
  laptop. Each gate uses one platform.
- **Novelty.** A short literature check found no matrix (propagator-based)
  PINN residual weight. It is not a full review (`docs/progress.md`).
