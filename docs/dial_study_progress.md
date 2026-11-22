# Dial study: running record

This log follows [`dial_study_plan.md`](dial_study_plan.md). It records each
step, check and deviation in order. Results are recorded only after the
corresponding step has finished.

## 22 November 2026

### Plan
The plan was committed (`ab63a28`) before any code for the study was written
and before any of its PINNs was trained.

### Amendment before training
Two training settings were changed before any training. They are described in
plan section 5.

- **Soft initial condition with lambda_IC = 100**, instead of the hard form.
  With x_hat = x0 + (t/T)(g − x0), a fast transient forces the raw output g to
  reach about T · kappa · d near t = 0. For kappa = 64 and d = 1 that is about
  128, which would make the stiff settings mostly a test of representing that
  spike.
- **Output scaling from all nine reference windows**, not only window 0.

Neither change uses any PINN outcome.

### Code
- **The PINN code now accepts any system with `rhs_torch`.** `src/pinn/train.py`
  gained `system_rhs`, and `src/pinn/runs.py` gained `job_system`. The Lorenz
  path is unchanged, and Turinici time weighting stays Lorenz-only.
- **New systems.** `src/systems/dial_systems.py` defines the stiff linear saddle
  (family L) and the Davis–Skodje system (family D) at the planned settings.
- **Integrated propagator.** `src/dynamics/tangent_dynamics.py` gained
  `integrated_propagators`, which computes K(t; dt).
- **New tests** in `tests/test_dial_systems.py` cover:
  - eigenvalues and directions;
  - the exact solution against DOP853;
  - the Jacobian against finite differences;
  - slow-manifold invariance;
  - agreement between the torch and NumPy right-hand sides;
  - a zero residual for the exact solution;
  - training and evaluation on the new systems;
  - stacked against separate training;
  - the closed form of K for a linear system.
- **Full suite:** 147 passed.

### Reference step
`python experiments/24_dial_reference.py` computed reference trajectories,
step propagators, singular directions and integrated propagators for all eight
settings and windows 0–8, on the grid h = 0.002 over [0, 2]. Output is in
`results/dial_study/reference/`.

**Checks.**
- **Family D:**
  - DOP853 agrees with a ten-times-looser run within 1e-8;
  - the propagator agrees with finite differences of the flow within 1e-4
    relative.
- **Family L:** the exact formula is the reference, and DOP853 only
  cross-checks it. On the stiffest setting (kappa = 64) the explicit solver is
  itself accurate only to about 2e-8 relative. The cross-check threshold was
  therefore set to 1e-7 instead of the first value, 1e-9, which only that
  setting missed. After that, all checks passed.

**Dial descriptives at dt = 0.1, windows 1–8** (dynamics only, no PINN):

| setting | dial | median sigma_1/sigma_2 | median integrated lever |
|---|---:|---:|---:|
| saddle, kappa = 1 | 2 | 1.22 | 1.11 |
| saddle, kappa = 4 | 5 | 1.65 | 1.28 |
| saddle, kappa = 16 | 17 | 5.47 | 2.11 |
| saddle, kappa = 64 | 65 | 665 | 6.73 |
| Davis–Skodje, gamma = 3 | 2 | 1.25 | 1.11 |
| Davis–Skodje, gamma = 6 | 5 | 1.71 | 1.29 |
| Davis–Skodje, gamma = 18 | 17 | 5.95 | 2.14 |
| Davis–Skodje, gamma = 66 | 65 | 746 | 6.83 |

For comparison, the October Lorenz set has a median sigma_1/sigma_2 of 1.8 at
dt = 0.1. The two lowest settings of each family are therefore Lorenz-like, and
the top setting separates directions several hundred times more strongly.

The integrated lever is the ratio of K's singular values. It is the most that
direction can change the error created by a residual that stays constant over
the horizon, so it is much smaller than the single-push ratio sigma_1/sigma_2.

### Pilot (accuracy only)
The pilot was started at 20,000 iterations: window 0, seed 0, at kappa = 64 and
gamma = 66. Its result is recorded below once it finishes.

**Pilot result.** `experiments/25_dial_pinn_runs.py --pilot` trained window 0,
seed 0 at the two stiffest settings on the CPU. Each run took about 132 s.

| setting | max error | relative to max state norm | median error | pilot rule (≤ 5%) |
|---|---:|---:|---:|---|
| saddle, kappa = 64 | 0.350 | 4.74% | 0.117 | pass |
| Davis–Skodje, gamma = 66 | 0.00078 | 0.05% | 0.00040 | pass |

Both pilots passed, so the main runs use 20,000 iterations as planned. No
direction statistic was computed in the pilot. Output is in
`results/dial_study/pilot/`.

### Main runs
- **Started:** at about 18:05 IST.
- **Devices.** The plan does not fix the device. To halve the wall time:
  - family L trains on the GPU, as one stacked model of 40 runs per setting;
  - family D trains on the CPU, with ten processes and one run each.
- **Why the device split does not matter.** Stacked and separate training give
  the same models up to floating-point rounding (tested in
  `tests/test_dial_systems.py`), and each family is analysed separately.
- **An aborted first launch.** A single GPU process for all settings was
  stopped after about two minutes, before it had saved any run, and replaced
  by this split.
- **Device change for family D (about 18:37 IST).** On this machine the
  ten-process CPU run was much slower than the single-run pilot: 20 of the
  first setting's 40 runs took about 30 minutes, which projected to about four
  hours for the family. The likely cause is that six physical cores were
  shared by ten processes. The CPU job was stopped and its 20 finished runs of
  gamma = 3 were set aside unused. Family D is retrained on the GPU after
  family L, as one stacked model per setting, so all 320 runs share one device
  and training method. No outcome of the set-aside runs was looked at.

### Training finished
- **Family L:** about 18:06–19:05 IST on the GPU, at 462–900 s per setting.
  The first two settings shared the machine with the CPU job.
- **Family D:** about 19:05–19:35 IST on the GPU, at 441–475 s per setting.
- **All 320 runs are finite.** `results/dial_study/pinn_summary.json` lists
  each run's accuracy.
- **Runs above 5% maximum error: 16.**
  - Six saddle runs at kappa = 64 reached 5.0–6.6% of the maximum state norm.
  - Ten Davis–Skodje runs at gamma = 3 and 6, on windows 7 and 8, missed the
    soft initial condition, with an initial error of 1.1–3.9.
  - Under the plan all runs are kept.

### Analysis
`python experiments/26_dial_analysis.py`, output in
`results/dial_study/analysis/`.

**Primary results** (dt = 0.1, target N, gain of M2 over B):

| setting | gain | verdict |
|---|---|---|
| saddle, kappa = 1 | +0.005 | fails |
| saddle, kappa = 4 | +0.001 | fails |
| saddle, kappa = 16 | +0.026 | inconclusive |
| saddle, kappa = 64 | +0.092 | works |
| Davis–Skodje, gamma = 3 | −0.000 | fails |
| Davis–Skodje, gamma = 6 | +0.003 | fails |
| Davis–Skodje, gamma = 18 | +0.072 | works |
| Davis–Skodje, gamma = 66 | +0.128 | inconclusive |

**Planned verdicts.**
- Family L is supported (Kendall's tau 0.67).
- Family D is not supported: its largest setting failed the time-shuffle
  condition. The shuffled gain was 0.129 against 0.128.
- Overall: **partly supported**.

**Observations recorded after the result** (none of them changes the planned
verdict):
- **The shuffle control is degenerate at large gamma.** Within a run, s₁ turns
  by about 11°, 8°, 3° and 0.0° at gamma = 3, 6, 18 and 66. A time shuffle of a
  constant vector cannot differ from it. The plan had excluded the shuffle for
  family L for the same reason.
- **Late-time sensitivity.** This was added after seeing family L's result and
  before family D's analysis. With samples at t ≥ 0.25 only, the gains are
  +0.076 (saddle, kappa = 64), +0.078 (gamma = 18) and +0.134 (gamma = 66).
- **Accurate runs only.** Leaving out the 16 runs above 5% maximum error gives
  −0.001 (gamma = 3), +0.014 (gamma = 6) and +0.097 (kappa = 64).
- **Figure fix.** The x-axis tick labels of the dial figure overlapped. Only
  the tick labels were changed and the figure redrawn; no number changed.
- **Family L was looked at early.** Its analysis was run once while family D
  was still training, in order to check the pipeline on real data. Nothing in
  family D's training or the analysis was changed afterwards, apart from the
  added late-time sensitivity, which is reported for all settings.

The write-up is in [`dial_study_results.md`](dial_study_results.md).
