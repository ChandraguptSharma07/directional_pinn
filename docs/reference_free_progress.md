# Reference-free directions: running record

This log follows [`reference_free_plan.md`](reference_free_plan.md).

## 22 November 2026

- **Plan.** Committed as `bf1063b` before any reference-free direction was
  computed and before any fresh-window PINN was trained.
- **Code.**
  - `src/systems/dial_systems.py` gained the fresh windows 9–16.
  - `src/dynamics/tangent_dynamics.py` gained `pinn_path_step_propagators`
    (estimate P) and `frozen_jacobian_propagators` (estimate F).
  - Experiments 24 and 25 accept `--settings`, `--windows` and
    `--reference-dir`.
  - New tests check that P computed on the exact path reproduces the reference
    propagators, and that F is exact for the linear saddle.
- **Fresh references.** `experiments/24_dial_reference.py` computed the windows
  9–16 for the four planned settings into `results/reference_free/reference/`.
  All checks passed.
- **Fresh PINNs.** Training started on the GPU at about 20:35 IST: 160 runs, one
  stacked model per setting, the same settings as the dial study.
- **Integrator for P changed before any fresh result was seen.**
  - **What failed.** A first development-data run of the analysis script used
    an adaptive DOP853 tangent integration along the PINN path. It stopped
    with "required step size is less than spacing between numbers" on one
    Davis–Skodje run (gamma = 3, window 7, seed 4). That run missed its initial
    condition, and its PINN path goes down to y1 = −5.3, through the
    singularity of the equations at y1 = −1.
  - **The new integrator.** P now uses the exponential midpoint rule,
    expm(J(x_hat(t_i + h/2)) h) for each 0.002 step. The rule is second order
    and exact for a constant Jacobian. On the exact path it reproduces the
    reference propagator over dt = 0.1 to a relative error below 1e-5
    (tested).
  - **Dropped samples.** Samples whose reference-free propagator is not finite
    are dropped from all models and counted in the metrics.
  - **This is an implementation detail of the planned P,** decided after
    looking only at development data. No fresh-window result had been computed
    at that point.
