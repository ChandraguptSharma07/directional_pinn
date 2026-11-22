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
