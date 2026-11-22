# Turinici adaptive time-weighting continuation — 22 November 2026

## Goal and source audit

The user asked to implement and test the reference paper's adaptive time-weighting method, then update the final LaTeX report. The official source repository was cloned to `/tmp/turinici_time_sampling` at commit `0595f4c526a82d651800bcd98f2d4f2c2f995bc3` from <https://github.com/gabriel-turinici/physics_informed_neural_networks_time_sampling>. Its `pinn_lorenz.py` uses TensorFlow and a fixed grid of 256 times on a half-unit Lorenz interval. The `lyapunov_lambda` branch estimates a scalar rate as `f(U)·U/(1e-10+||U||²)`, exponentiates a cumulative rate to produce normalized time weights, and updates weights after every optimizer epoch. It enforces the initial condition exactly by shifting the network output. Our existing trainer is PyTorch with a different architecture and a hard-IC option, so a matched in-project port is needed for an interpretable test.

The paper's code sets `timestep=1/(N-1)` even when the Lorenz interval is `T=0.5`; the paper's Eq. (13) is expressed in physical time. Our port will use `T/(N-1)` to implement that equation on our interval and record the distinction. The resulting experiment tests the paper's weighting *rule within our model*, not a byte-for-byte reproduction of its entire TensorFlow training setup. We will compare it with uniform weighting on the same fixed grid, initial condition, model, seed, iteration count, and optimizer schedule. This log will be updated with each completed run and the result before the session ends.

## Implementation and verification

Added `time_weighting="turinici"` and a fixed-grid collocation option to `src/pinn/train.py`. The state-based rate is computed from the current model after each Adam update; normalized exponential weights are held fixed during the next residual-gradient update. The implementation uses Eq. (13)'s physical-time spacing, stable softmax normalization, and a `1e-10` denominator as in the author's code. Existing training remains the default. The adaptive option currently supports the single-model trainer, which is sufficient for the matched pilot. The focused PINN suite passed **21/21** including the new weight formula, normalization, short adaptive training, and invalid-configuration check.

## Run 1: matched uniform fixed-grid control

`experiments/11_hard_ic_pilot.py --iterations 20000 --collocation 256 --collocation-scheme fixed --time-weighting uniform --out results/adaptive_time_weighting/w0_t2_uniform_fixed` finished with exit 0 in 54.0 seconds. This hard-IC, window-0, seed-0 global `T=2` run had median trajectory error **17.4733**, maximum error **40.6466**, error below 1 on **3.20%** of the window, median residual norm **0.0519**, and final training loss **25.4405**. It is the fixed-grid control for the adaptive run, not a repeat of the earlier stratified/curriculum hard-IC pilot.

## Run 2: adaptive fixed-grid weighting

`experiments/11_hard_ic_pilot.py --iterations 20000 --collocation 256 --collocation-scheme fixed --time-weighting turinici --out results/adaptive_time_weighting/w0_t2_turinici_fixed` finished with exit 0 in 59.9 seconds. Median trajectory error was **17.4149**, maximum **40.6166**, fraction below error 1 **4.10%**, median residual **0.1128**, final adaptive weighted loss **1.5336**. Against the matched uniform run, this is only a 0.0584 reduction in median trajectory error; both are unusable global `T=2` solves. The weighted and uniform final losses have different objectives and must not be compared as if they measured the same unweighted error.

The final normalized weights have a max/min contrast of **201.9**. Only **1.34%** of total weight lies on the first quarter of grid points, while **74.10%** lies on the last quarter. The paper's state/zero rate proxy therefore emphasized *late* times in this trained pilot, even though Lorenz's long-run leading Lyapunov exponent is positive. This is an observed behavior of the proxy and model, not a proof that the paper's theoretical density is wrong; Eq. (13) requires a rate that bounds error growth, whereas the practical Eq. (17) estimator is only approximate.

## Next matched test

Because the paper's Lorenz experiment uses short solves, the next test applies the same adaptive rule on our existing eight consecutive `0.25`-unit hard-IC segments and compares it with a uniform fixed-grid sequential control. Both will keep segment count, seed, grid size, update count, and optimizer schedule identical. The existing stratified piecewise result remains a separate benchmark.

## Run 3: matched sequential uniform fixed-grid control

`experiments/13_piecewise_pinn_pilot.py --window 0 --seed 0 --collocation-scheme fixed --time-weighting uniform --out results/adaptive_time_weighting/w0_t2_piecewise_uniform_fixed` completed in 87.3 seconds. Eight `0.25`-unit segments, 4000 updates and 256 fixed collocation points per segment gave median trajectory error **0.03594**, maximum **0.3130**, final error **0.3130**, and **100%** of times below error 1. This confirms that the fixed-grid sequential control is already a usable trajectory on the pilot condition. The matched adaptive sequential run is underway.

## Run 4: matched sequential adaptive pilot

The window-0 adaptive run completed in 97.4 seconds. Median error was **0.04203**, maximum and final error **0.4094**, median residual **0.4164**, and **100%** of times remained below error 1. The uniform fixed-grid control had median **0.03594**, maximum **0.3130**, and median residual **0.4068**. Adaptive weighting was slightly worse on this pilot initial condition, although both models were accurate by the error-1 criterion. The earlier stratified sequential pilot's median 0.0259 is a different sampling setup and is not the matched control.

## Eight-window seed-0 extension

To avoid judging the method on a single pilot condition, `experiments/22_turinici_weighting_study.py` ran both fixed-grid methods on all eight analysis windows at seed 0. It saved each trajectory and updated `results/adaptive_time_weighting/sequential_seed0_summary.json` after every completed job, so the study could resume after interruption. These original analysis windows were previously seen during project development, and one seed per window gives only descriptive evidence. The first completed pair, window 1, showed both methods failing late on that difficult initial condition.

### Completed eight-window comparison

All **16/16** runs finished with saved trajectories and metrics. No starting condition was omitted. Each row below is one starting condition with seed 0; both methods used eight 0.25-unit hard-IC segments, 4000 Adam steps and 256 fixed collocation times per segment.

| Window | Uniform median error | Adaptive median error | Uniform max error | Adaptive max error |
|---:|---:|---:|---:|---:|
| 1 | 0.04474 | 0.04331 | 7.3613 | 7.6281 |
| 2 | 0.05781 | 0.06783 | 0.4514 | 0.5184 |
| 3 | 0.04615 | 0.05122 | 0.1962 | 0.2536 |
| 4 | 0.02412 | 0.02437 | 0.0843 | 0.1050 |
| 5 | 0.04291 | 0.05001 | 0.1492 | 0.1604 |
| 6 | 0.04362 | 0.05260 | 0.2637 | 0.3181 |
| 7 | 0.02612 | 0.02718 | 0.0672 | 0.0685 |
| 8 | 0.03860 | 0.04294 | 0.1398 | 0.1695 |

The median across the eight run-level median errors is **0.04326 uniform** versus **0.04666 adaptive**. Adaptive has the lower median error on **1/8** windows; each method stays below error 1 throughout on **7/8** windows. Mean runtime per two-unit trajectory was **88.2 s uniform** versus **96.9 s adaptive** on this machine. The adaptive rule therefore did not improve this one-seed sequential comparison; the difference is small relative to the already accurate trajectories, and no population-level or cross-system claim follows from eight previously used windows. This is consistent with the pilot's near-uniform weights on short segments (final max/min ratios about 1.4–2.3). The global failure and this sequential result support the narrower interpretation that the paper's theoretical weighting rule, as ported here, was not the main source of our observed model improvement; segmentation remains the effective change in the studied setup.

The full automated test suite passed **131/131** after implementing the weighting option. The numerical source of every row is `results/adaptive_time_weighting/sequential_seed0_summary.json`, and each run has a saved configuration, history, trained segment weights, and trajectory arrays. The final LaTeX report will include the method equations, source-code provenance, matched results, and these limits.

## Literal source-code time-scale sensitivity

The author's Lorenz code sets the cumulative-rate step to `1/(N-1)` even though its physical interval is `T=0.5`; our first port used `T/(N-1)` from paper Eq. (13). To make the source comparison explicit, added `time_weighting="turinici_code"`, which changes only that step in the same adaptive weight update. Its formula check passes. A matched global pilot and all eight seed-0 sequential windows were run with this literal scaling, using the already saved uniform fixed-grid controls. This sensitivity is separate from the theoretical-equation implementation above and is reported regardless of outcome.

### Completed literal-scaling sensitivity

The global `T=2` source-scaling pilot finished in 59.7 seconds with median trajectory error **17.4887**, maximum **40.6455**, and error below 1 for **3.30%** of the window. The matched uniform control had median **17.4733**, maximum **40.6466**, and **3.20%** below 1. Neither solved the global interval.

All **8/8** seed-0 sequential source-scaling runs finished; their summary is `results/adaptive_time_weighting/sequential_code_seed0_summary.json`. The per-window median/max errors were: window 1 **0.13425/36.8641**, 2 **0.13275/0.8259**, 3 **0.09457/0.5071**, 4 **0.07770/0.4043**, 5 **0.09132/0.3598**, 6 **0.23774/1.4802**, 7 **0.05422/0.1138**, 8 **0.11110/0.4789**. The median of run-level medians is **0.10283**, compared with **0.04326** for uniform and **0.04666** for the Eq. (13) scaling. Source scaling lowered median error on **0/8** windows relative to uniform; it stayed below error 1 throughout on **6/8** (uniform 7/8, Eq. (13) 7/8). Mean runtime was **98.5 s** per trajectory versus 88.2 s uniform. On window 1, final within-segment weight contrasts ranged from about 3.7 to 632, much stronger than the Eq. (13) pilot's roughly 1.4–2.3. This observation is consistent with a more aggressive weighting distribution, but it does not establish the unique cause of poorer trajectory accuracy.

These experiments evaluate two faithful interpretations of the author's stated method within our different PyTorch architecture and hard-IC construction. They do **not** re-run the author's TensorFlow model, published architecture, or its complete benchmark. The paper's minimax result concerns a bound under its assumptions; neither this code sensitivity nor the lack of benefit in our training contradicts that theorem. It shows that adaptive time weights were not a demonstrated improvement for this project's Lorenz solver under matched budgets.
