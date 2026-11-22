# Matrix-weighting study: running record

This log follows [`plan.md`](plan.md). Each entry is written after its step
has finished.

## 22 November 2026

### Plan

The plan was committed (`6fb0845`) before any code for the study was written
and before any of its networks was trained.

### Code

Everything is in `matrix_weighting/`.

- `mw/systems.py` defines the systems:
  - the dial study's linear saddle and Davis–Skodje system, copied rather than imported;
  - Lorenz-63, optionally driving k damped modes.

  Each system has batched right-hand sides and Jacobians in NumPy and in
  PyTorch.
- `mw/model.py` holds a batch of independent 4 × 64 tanh networks.
  - The initial condition is imposed exactly.
  - The exact time derivative is carried forward through the layers.
  - Each network's initialization depends only on its (seed, initial
    condition) pair.
- `mw/weights.py` builds the six weights of the plan, plus the final-time
  and causal side variants:
  - the step propagators;
  - the trapezoidal window Gramian recursion;
  - the final-time Cauchy–Green tensor;
  - the normalizations.
- `mw/train.py` holds the training loop and the trajectory metrics.
- `mw/reference.py` holds the references (DOP853 at rtol = atol = 1e-12, or
  the exact saddle formula) and the Gate 1 attractor states.
- `experiments/run_job.py`, `experiments/pool.py` and
  `experiments/make_jobs.py` handle jobs, worker pools and the job files.
- `remote/kaggle_kernel.py` builds a private Kaggle script kernel. The kernel
  contains the code and job files; no credentials are written into it.

### Tests

`python -m pytest tests` in `matrix_weighting/`: **26 passed**. The tests
cover:

- **Systems:**
  - Jacobians against finite differences, for every system;
  - NumPy against PyTorch;
  - the saddle formula against DOP853.
- **Model:**
  - the forward-mode time derivative against autograd;
  - the exact initial state;
  - an initialization that does not depend on batch companions;
  - batched against separate training (identical to 1e-9).
- **Weights:**
  - the Gramian recursion against the closed form for scalar growth;
  - the Gramian and the final-time tensor against matrix exponentials for the saddle;
  - the weight normalizations;
  - the trace inequality behind the minimax weight;
  - Turinici weights identical to the repository's port.
- **Training:** a short run of every method.

### Frozen initial conditions

`make_jobs.py gate1` wrote `results/gate1/initial_conditions.json` (12 attractor
states, seed 2026112201) before any Lorenz training. All job files of Gates
0–2 and the side runs were written by `make_jobs.py` before any of their
training and are committed under `jobs/`.

### Timing check (replaces the plan's window-0 pilot)

Three gate jobs were run for **300 of 20,000 steps** on nixxa to measure speed:

- Lorenz T = 2 with `U` (36 networks): 0.093 s per step on one thread;
- Lorenz T = 2 with `M-self` (36 networks): about the same;
- saddle κ = 64 with `M-orc` (24 networks): 0.062 s per step.

Their 300-step errors were printed in the logs but used for nothing. No
setting was changed because of them. The window-0 pilot jobs in `jobs/pilot`
were not run.

### Amendment: where each gate runs (before any gate result)

The plan put Gates 0 and 1 on nixxa. nixxa is a 16-thread laptop with 15 GB of
memory, and its owner was using it at the same time.

- **First attempt:** 15 single-thread jobs. This left under 1 GB free, with
  no swap, so it was stopped within minutes.
- **Second attempt:** 8 jobs of 2 threads. This slowed each Lorenz job to
  0.24 s per step, so Gates 0 and 1 together would have needed about 7
  hours.

Both attempts were deleted before any job finished.

**New allocation:**

- **Gate 1 (Lorenz-63, the main test):** runs alone on nixxa, 10
  single-thread workers at nice 10.
- **Gate 0:** moved to Kaggle (GPU).
- **Gate 2 and the side runs:** stay on Kaggle as planned.

**Each gate runs on one platform**, so every paired comparison is between
networks trained on the same hardware. Only where jobs run has changed:
settings, seeds, initial conditions and decision rules are unchanged.

Kaggle kernels (private, GPU), each running its jobs with three or four
concurrent workers per GPU:

| account | kernel | jobs |
|---|---|---|
| chandraguptsharma | `mw-g2-k29` | Gate 2, k = 29 (6) |
| duoshingaming | `mw-g2-k9-k0` | Gate 2, k = 9 and k = 0 (12) |
| duoshingaming | `mw-gate0-dial` | Gate 0, all (40) |
| hy1069 | `mw-g2-k3-final` | Gate 2, k = 3 (6); side, final-time weights (6) |
| yess16 | `mw-side-causal-bench` | side: causal (9), Turinici benchmark (6) |
| nikita245678k | `mw-side-sens` | side: sensitivity (12) |

### Added side run: GPU replicate of Gate 1 (before any Gate 1 result)

The 18 Gate 1 jobs, unchanged except for the output folder
(`results/side/gate1_gpu/`), were queued on Kaggle GPUs as an exploratory
replicate (`jobs/side_gate1gpu`, kernels `hy1069/mw-g1-gpu-a` and
`yess16/mw-g1-gpu-b`). It checks whether the Gate 1 conclusion survives
different hardware and numerics. The planned Gate 1 decision uses only the
nixxa runs.

### Literature check (required by plan section 11)

Searches on 22 November 2026 found no PINN loss weighted by a tangent-linear
(propagator) matrix. The searches combined Lyapunov vectors, tangent-linear
sensitivity, adjoint and dual weighted residuals, shadowing, and PINN loss
weighting. The closest work:

- [Turinici, arXiv:2407.21642v2](https://arxiv.org/html/2407.21642v2):
  scalar exponent-based time weights; the direction-blind special case.
- [arXiv:2604.01835](https://arxiv.org/abs/2604.01835): a
  dual-weighted-residual error estimator, localized to choose PINN
  collocation points. It is a scalar indicator per point for a chosen goal,
  not a matrix weight in the loss.
- [Wang, Sankaran and Perdikaris, arXiv:2203.07404](https://arxiv.org/pdf/2203.07404):
  causal time weights.
- [Hillebrecht and Unger, arXiv:2203.17055](https://pdf.arxiv.org/pdf/2203.17055):
  a posteriori ODE error bounds for PINNs.
- [De Ryck et al., arXiv:2310.05801](https://arxiv.org/pdf/2310.05801):
  operator preconditioning of physics-informed training.
- [Özalp et al., arXiv:2305.15111](https://arxiv.org/pdf/2305.15111v1):
  covariant Lyapunov vectors used to evaluate physics-informed LSTMs, not to
  weight a loss.

This is a quick check, not a full review. A novelty statement still needs a
fuller search, for example of the data-assimilation literature on weighting
in the unstable subspace.

### GPU out-of-memory in the direction-blind weight; fix (22 November, evening)

On Kaggle GPUs, `torch.linalg.eigvalsh` asked for about 9 GB of workspace for
one batch of 36 × 512 small symmetric matrices. Three `S-self` jobs ran out of
memory at their first weight update and wrote no results:

- side GPU replicate: Lorenz T = 0.5 and T = 1;
- Gate 2: k = 0.

`mw/weights.py::best_scalar` now computes these eigenvalues on the CPU. The
mathematics is unchanged and the tests still pass (26/26). Jobs that finished
before the fix are kept. Failed jobs are rerun with the fixed code on the
same kind of GPU (kernel `hy1069/mw-rerun-1`). The nixxa runs use the CPU and
were never affected.

### First Kaggle results (exploratory side runs and part of Gate 2)

**Status.** These finished first. No gate decision is drawn here. Gate 2 waits
for k = 29, and Gate 1 is still running on nixxa.

**GPU replicate of Gate 1 (side, T4).** Median log10 E_rms over 36 networks,
with the success rate (E_max < 1 over the whole window):

| T | `U` | `TUR` | `S-orc` | `M-orc` | `M-self` |
|---|---|---|---|---|---|
| 0.5 | −3.05 (36/36) | −3.01 (36/36) | −3.02 (36/36) | −2.93 (36/36) | −2.88 (36/36) |
| 1 | 1.29 (7/36) | 1.29 (5/36) | 1.27 (9/36) | −2.36 (23/36) | −2.29 (27/36) |

`S-self` is missing because of the out-of-memory failure; it is being rerun.

**Gate 2 so far (Lorenz-part error, T = 1).** Success rates (E_max < 1),
36 networks each:

| method | k = 0 | k = 3 | k = 9 |
|---|---|---|---|
| `M-self` | 75% | 75% | 75% |
| `S-self` | rerunning | 56% | 19% |
| `M-orc` | 64% | 47% | 28% |
| `U` | 19% | 19% | 8% |
| `S-orc` | 25% | 14% | 11% |
| `TUR` | 14% | 11% | 0% |

The k = 29 runs are still queued.

**Time profiles (from the training histories).**

- **The matrix and scalar weights spread over time almost identically.**
  Over Lorenz T = 1, the per-time weight tr W / n falls from about 1.7 at
  t = 0 to 0.06–0.10 at t = T for both `M-orc` and `S-orc`. They differ only
  in how each time's weight is split across directions. Yet only `M-orc`
  escapes the failure plateau (median grid error 0.0044 against 18.4).
- **`M-self` is not a hidden causal schedule.** Its time profile settles to
  the oracle's within about 2,000 steps.
- **`TUR` puts more weight late in the window, not early.** Its weight rises
  from about 0.26 at t = 0 to 2.75 at t = T. Its growth proxy ⟨f(x), x⟩/‖x‖²
  is often negative on Lorenz windows.

**Mechanism check** (`experiments/mechanism.py`; oracle Gramian along the
reference, so diagnostic only).

- **T = 0.5, where every method succeeds:**
  - Residual energy along the least harmful direction of G: 37–43% with the
    matrix weights, against 17% (`U`) and 22% (`TUR`).
  - Harm per unit residual: about 0.19 against 0.35.
  - Final errors stay comparable (median E_rms 0.9e-3 to 1.3e-3).
- **T = 1:**
  - The successful matrix-weighted networks leave 60–73% of their residual
    along the least harmful direction.
  - The failed `U`, `TUR` and `S-orc` networks put about 60–70% along the
    most harmful one.

The matrix weight moves the residual into the directions that the dynamics
erase, as the theory predicts.

### Two kernels moved (still before any Gate 0 or k = 29 result)

`duoshingaming/mw-gate0-dial` and `chandraguptsharma/mw-g2-k29` stayed
queued for over 30 minutes while other accounts' kernels started at once.

- The same job files, now with the eigenvalue fix, were pushed as
  `hy1069/mw-gate0-dial-b` and `nikita245678k/mw-g2-k29-b`.
- The queued first copies were replaced by a no-op version so they cannot
  run as well.
- Gate 0 and the k = 29 results come from the `-b` kernels.

### Gate 3 and side runs queued

- **Gate 3 (plan amendment 1):**
  - Rössler: `chandraguptsharma/mw-gate3-rossler`, running.
  - Lorenz-96 and the direction-only side run: `duoshingaming/mw-gate3-l96-dir`, queued.
- **Time-marching side run (plan amendment 2):** built, waiting for a free
  Kaggle GPU slot. Every account is at its limit of 2 concurrent GPU
  sessions.

### Rerun kernel: two jobs recovered, one lost to a packaging bug

`mw-rerun-1` produced the Gate 2 k = 0 `S-self` job and the replicate's
T = 0.5 `S-self` job. Its third job, the replicate's T = 1 `S-self`, was
silently dropped. Two job files from different folders share the name
`lorenz63_T1_S-self.json`, and the packer stored both under that name.

`remote/kaggle_kernel.py` now prefixes each packed job file with its folder
name. No earlier kernel mixed folders with clashing names. The dropped job
goes in `mw-rerun-2` when a GPU slot frees.

### Sensitivity side run (Lorenz T = 1, exploratory)

`nikita245678k/mw-side-sens` finished 9 of 12 jobs. Its three `S-self` jobs
hit the eigenvalue out-of-memory failure (old code) and were added to
`mw-rerun-2`. Median log10 E_rms over 36 networks:

| variant | `U` | `TUR` | `M-self` |
|---|---|---|---|
| 50,000 steps (2.5 times the budget) | 1.29 | 1.29 | −2.65 |
| width 128 | 1.29 | 1.29 | −2.45 |
| grid of 1,024 times | 1.29 | 1.29 | −2.25 |

The uniform and Turinici failures at T = 1 do not go away with 2.5 times more
steps, a wider network or a denser grid. The `M-self` result holds under all
three changes.

### Rerun results change the reading of Lorenz T = 1 (exploratory)

`mw-rerun-2` and the earlier reruns completed the missing `S-self` jobs.
Medians of log10 E_rms:

- **On plain Lorenz-63 at T = 1, `S-self` does about as well as `M-self`.**
  - Gate 2, k = 0: −2.27 (67% success) against −2.29 (75%).
  - GPU replicate: −2.27 against −2.29.
  - Sensitivity variants: −2.96, −2.47 and −2.67, against −2.65, −2.45 and
    −2.25 for `M-self`.
- **The oracle scalar weight `S-orc` fails** (25% success), although its time
  profile is the one `S-self` ends with.

**The training histories (post hoc) suggest why.** For its first ~3,000 steps,
`S-self` puts very little weight on the second half of the window. Its weight
at t/T = 0.75 is about 0.09, against 0.46 for the oracle; at t = T it is about
0.004, against 0.06. Once the first half is right, its profile relaxes to the
oracle's. So the self-consistent weights act like an automatic **causal
schedule**: the late window counts for little while the network's own path
is still wrong there.

**The present reading has two separate effects:**

1. **A better time weight.**
   - The Gramian along the network's own path, even as a scalar, rescues
     Lorenz T = 1: 19% success for `U`, 14% for `TUR`, about 67–75% for the
     self-consistent Gramian weights.
   - Turinici's state-based proxy does not.
2. **Direction:**
   - It matters with fixed (oracle) weights: `M-orc` reaches 64% against 25%
     for `S-orc`.
   - It matters with many contracting directions: in Gate 2 at k = 3 and 9,
     `M-self` reaches 75% against 56% and 19% for `S-self`.
   - It matters on Rössler at T = 10: about 25 times lower error than
     `S-self`.
   - On plain Lorenz-63 at T = 1 it adds little once the weights are
     self-consistent.

This is exploratory. The planned H2 decision on Lorenz-63 comes from the
nixxa Gate 1 runs. If `S-self` matches `M-self` there too, H2 will be
"partial", and that is what will be reported.

### Gate 0 complete: H1 partial

`hy1069/mw-gate0-dial-b` finished all 40 jobs (24 networks each, T4 GPUs).
The planned analysis (`results/gate0/analysis.md`) gives **H1: partial**.

**Linear saddle.**

| κ (dial) | `M-orc` against `S-orc` |
|---|---|
| 1 (2) | worse, ×1.9 higher error |
| 4 (5) | worse, ×3.1 |
| 16 (17) | worse, ×4.3 |
| 64 (65) | better, ×0.50; also better than `U` (×0.17) and `TUR` (×0.30) |

**Davis–Skodje.** `M-orc` is worse than `S-orc` at γ = 3, 6 and 18 (×1.2 to
×2.5) and the same at γ = 66. The self-consistent versions behave like the
oracle ones.

**Reading.** In these two-dimensional problems, nearly every method succeeds
(E_max < 1 in 24/24 runs; the only failures are `U` and `TUR` at κ = 64).
The final errors are small, 1e-2 to 1e-4. There, the matrix weight costs some
accuracy, because it accepts a larger residual along the contracting
direction. It helps only in the hardest setting, the κ = 64 saddle, where the
plain loss struggles. So the matrix weight is not uniformly better: its value
lies in the hard regime, where standard training fails or nearly fails.

### Gate 2 complete: H3 not supported; the direction effect grows with k

`nikita245678k/mw-g2-k29-b` finished the k = 29 jobs. Planned analysis on the
Lorenz-part error (`results/gate2/analysis.md`):

**H3 is not supported** by the planned rule. The gain of `M-self` over `U` is
already very large at k = 0 (mean Δ −2.16, about 145 times lower error) and
does not grow reliably with k. The gain at k = 29 minus the gain at k = 0 is
+0.28, with interval [−0.30, +0.98].

**Secondary contrasts** were computed by the planned script but do not decide
H3. The gain of `M-self` over the direction-blind `S-self` grows steadily
with the number of contracting directions:

| k | Δ, `M-self` against `S-self` | interval | fold change | verdict |
|---|---|---|---|---|
| 0 | −0.28 | [−0.93, +0.15] | ×0.52 | no decision |
| 3 | −0.89 | — | ×0.13 | beats |
| 9 | −1.69 | — | ×0.02 | beats |
| 29 | −2.01 | — | ×0.01 | beats |

This is the theory's prediction: a direction-blind weight wastes more of its
budget as more directions become harmless. With oracle weights, `M-orc` beats
`S-orc` at every k.

### nixxa memory (about 20:00)

The owner's Firefox grew, and free memory on nixxa fell to about 240 MB, with
no swap. Our 10 Gate 1 jobs were stable at about 3.9 GB. Their
`oom_score_adj`, and the pool's, was raised to 1000. If memory runs out, the
kernel then stops one of our jobs (to be rerun) rather than one of the
owner's programs. Training is unaffected.

### Lorenz-96 kernel moved again

`nikita245678k/mw-gate3-l96-dir-b` stayed queued for about 45 minutes. The same
job files went to `hy1069/mw-gate3-l96-dir-c`, and the queued copy was
replaced by a no-op. The Lorenz-96 results come from `-c`.

### Time-marching side run (plan amendment 2) complete

`hy1069/mw-side-sequential`: Lorenz-63 to horizon 4 from the 12 Gate 1
states, 3 seeds. Each segment had 20,000 steps.

| segment length | method | success (E_max < 1 on [0, 4]) | median log10 E_rms | GPU time |
|---|---|---|---|---|
| 0.5 (8 segments) | `U` | **27/36** | −1.49 | 1,641 s |
| 0.5 | `S-self` | 24/36 | −0.66 | 3,783 s |
| 0.5 | `M-self` | 25/36 | −0.85 | 3,655 s |
| 1.0 (4 segments) | `U` | 0/36 | 1.42 | 816 s |
| 1.0 | `S-self` | 4/36 | 1.24 | 2,741 s |
| 1.0 | `M-self` | 15/36 | −0.30 | 2,483 s |

**The expectation stated in advance held:** at segment length 1.0,
`M-self` succeeds in more runs than `U` (15 against 0).

**But the simplest option is still the most reliable:** uniform weighting
with short segments of 0.5 reaches the 4-unit horizon in 27/36 runs, with
the lowest error and the least GPU time.

**Interpretation.** The matrix weight makes longer windows trainable, but at
this budget fewer, longer segments do not beat more, shorter ones. Its
practical value is where a long single window is needed, not as a drop-in
replacement for time-marching.

### Gate 3 complete: H4 supported for Rössler and for Lorenz-96

`hy1069/mw-gate3-l96-dir-c` finished all 12 jobs. Planned analysis
(`results/gate3/analysis.md`) for Lorenz-96 (N = 8, F = 8): **H4 supported**.

| window | `M-self` vs `S-self` | vs `U` | vs `TUR` | success: `M-self` / `S-self` / `D-self` / `U` / `TUR` |
|---|---|---|---|---|
| T = 1 | ×0.41, beats | ×0.07, beats | ×0.02, beats | 36 / 36 / 35 / 29 / 13 of 36 |
| T = 2 | ×0.42, no decision | ×0.19, beats | ×0.16, beats | 10 / 1 / 0 / 0 / 0 of 36 |

Rössler had already given H4 supported. With Lorenz-96, the planned H4 rule
is met for both Gate 3 systems.

**Direction-only side run on Lorenz T = 1 (exploratory).**

- `D-self` (same total weight at every time; only the split across
  directions follows G): 23/36 successes, median log10 E_rms −1.56.
- `D-orc`: 10/36, median 1.21.

So direction alone already helps a lot when it is self-consistent.
Uniform weighting succeeds in 7/36 and Turinici's in 5/36.

### GPU replicate of Gate 1 at T = 2 (side)

`yess16/mw-g1-gpu-b` ran on one GPU and took 2.2 hours. At T = 2 every method
mostly fails: median log10 E_rms is between 1.15 and 1.40 for all six.

| method | `M-self` | `S-orc` | `S-self` | `M-orc` | `U` | `TUR` |
|---|---|---|---|---|---|---|
| successes (of 36) | 8 | 6 | 5 | 3 | 0 | 0 |

A single two-unit Lorenz window stays out of reach at this budget for every
weighting. `M-self` gives the most successes.

### Gate 1 on nixxa: nine jobs lost to the out-of-memory killer; checkpointing added

By 21:10 the pool reported 9 of its first 11 finished jobs as `FAILED (exit -9)`.

- **What happened.** The owner's programs left about 250 MB free. Our jobs had
  `oom_score_adj` 1000, so the kernel stopped them rather than the owner's
  programs, as intended.
- **When.** Eight of them had completed all 20,000 training steps. They were
  stopped in the final float64 evaluation, whose memory briefly rises. The
  ninth (T = 2 `M-orc`) was stopped during training.
- **What was lost.** No results were saved for the nine, and none were
  selected or dropped.
- **Jobs lost:**
  - T = 0.5: all six methods;
  - T = 2: `M-self`, `S-self` and `M-orc`.
- **Unaffected.** T = 1 `M-self` and `S-self` finished. Seven jobs (T = 1 and
  T = 2 `U`, `TUR` and `S-orc`, plus T = 1 `M-orc`) kept running.

**Fix (`mw/train.py`, tests 36/36).**

- The trained parameters and history are saved to `model.pt` (ignored by git)
  before evaluation. A job stopped during evaluation then resumes from the
  checkpoint instead of retraining. `summary.json` records
  `resumed_from_checkpoint`.
- The training graph and optimizer state are freed before evaluation.
- Evaluation runs in time chunks of 256 points. Each time is evaluated
  independently, so the values do not change.

**Rerun.** The nine lost jobs are queued on nixxa (`jobs/gate1_rerun` there:
same job files, 9 workers). They start automatically when the first pool
ends. Settings, seeds and initial conditions are unchanged.

### Preview: the Gate 1 rule applied to the GPU replicate (side run)

`analyze.py gate1_gpu` applies the H2 rule to `results/side/gate1_gpu` and
gives **partial**:

- `M-self` beats `U` and `TUR` at T = 1 (×0.01) and at T = 2 (×0.21, ×0.20).
- It does not beat `S-self` at any horizon (×1.10, ×0.52, ×0.59; every
  interval contains 0).
- At T = 0.5 nothing differs.

This is only a preview on other hardware; the planned decision waits for the
nixxa runs. The soft-initial-condition kernel stayed queued on hy1069. It was
moved to `nikita245678k/mw-side-soft-ic-b`, and the queued copy was replaced
by a no-op.
