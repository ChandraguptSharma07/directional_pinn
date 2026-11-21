# Interpolation follow-up: running record

This file records each work session and experiment run for the interpolation follow-up to the submitted project report. It separates completed checks from empirical Lorenz results. Update it before ending each work session.

## Session 1 — 21 November 2026, approximately 18:57 IST

### Aim

Test whether interpolating sparse PINN residual samples and averaging them over a trailing time window improves held-out prediction of future Lorenz-63 trajectory error. This is exploratory work after the original directional-alignment hypothesis was not supported.

### Work and run attempts

1. Searched the checkout and nearby project/home/temp directories for `trajectory.npz`, `residuals.npz`, and relevant `model.pt` files. No original PINN arrays or weights were found. `results/pinn_runs/runs/` contains JSON summaries and histories only; `results/lorenz_reference/data/` is also absent.
2. Inspected `experiments/06_pinn_runs.py`. It skips any run with an existing `metrics.json`, even if its large arrays are absent. Therefore regeneration must use a fresh `--out` directory, or the skip rule must be changed with care. A fresh directory preserves the published summaries.
3. The interpolation experiment `experiments/09_residual_interpolation.py` and helper `src/analysis/interpolation.py` were implemented in the preceding session. Synthetic end-to-end evaluation passed for 16 simulated runs, four horizons, and six candidate features per horizon. Compilation passed. A run against `results/pinn_runs` stopped cleanly with the expected missing-array message (80 missing files for 40 runs).
4. Created the ignored `.venv/` and installed NumPy, SciPy, Matplotlib, PyYAML, pandas, pytest, and CPU PyTorch. The sandbox initially could not resolve package hosts; approved retries downloaded the dependencies. On this Nix host the wheels also need `LD_LIBRARY_PATH` to include `/nix/store/2ga5nd1m56n5cx2wh8vbf6nrdhqk2f0q-gcc-15.3.0-lib/lib` for `libstdc++.so.6`; `MPLCONFIGDIR=/tmp/directional_matplotlib` avoids a nonwritable Matplotlib cache.
5. Ran `tests/test_interpolation.py` in the new environment: **3 passed**. Started rebuilding the Lorenz reference arrays into `results/interpolation_rebuild/reference/`, keeping the published result directory untouched.
6. Reference rebuild completed successfully: **36/36 checks passed**, all nine windows produced arrays, minimum trusted horizon 5.82 time units. Initial states `x0` for analysis windows 1–8 match the published `results/lorenz_reference/metrics.json` exactly (zero Euclidean difference).
7. Started `experiments/06_pinn_runs.py` with the rebuilt reference, fresh output `results/interpolation_rebuild/pinn_runs/`, CPU device, and two worker processes. This schedules **40 runs in 8 window stacks**. Two workers were chosen because the host reported about 2.9 GiB available memory when training started. The run is in progress; no outcome has been inferred yet.
8. The tool runtime was replaced at about 19:14 IST while the first two window stacks were still training. The session disappeared and no `trajectory.npz` or run `metrics.json` had been written. The reference arrays survived. This first PINN attempt produced **zero completed runs**. Restarting with a disk log and a detached process so a future tool-session reset does not discard progress.
9. A `nohup` background launch did not persist in this sandbox and produced no log or artifacts. Restarted the same 40-run, two-worker CPU command in a foreground tool session at about 19:45 IST, redirecting output to `results/interpolation_rebuild/pinn_train.log`. The log confirms `40 runs to train in 8 stacks on cpu`. The run is active; check the log and completed `trajectory.npz` count before inferring progress.
10. At about 19:56 IST, the foreground run remained active with two runnable worker processes and no error in the log. It had not completed a window stack, so the count of saved trajectory arrays was still zero. No interpolation score has been calculated.
11. At about 19:58 IST, windows 1 and 2 completed, producing **10/40** trajectory and residual arrays. Their median errors differ from the corresponding published summaries by at most 0.0011; the largest final-loss difference is 0.0063, consistent with a close rebuild under a different PyTorch version. The remaining six windows are still training.
12. Ran experiment 09 on the first 10 completed runs as an **end-to-end pipeline check only**, output `results/interpolation_rebuild/smoke_2_windows/metrics.json`. It completed and produced four horizon analyses. The leave-one-window-out scores and two-window bootstrap intervals are unstable; they are **not** an empirical conclusion and must not be compared with the published 8-window decision rule. The final analysis will use all 8 initial-condition windows and 40 runs.
    - Reconstruction check on these ten runs: mean relative error of the trailing-average magnitude is 0.0278 (linear) versus 0.0693 (sample-and-hold) for width 0.1, and 0.0216 versus 0.0501 for width 0.25. This checks numerical fidelity only; it does not establish predictive usefulness.
13. At about 20:10 IST, windows 3 and 4 also completed: **20/40** arrays are present. Across these 20 rebuilt runs, the largest absolute difference from the published median error is 0.059 and the largest final-loss difference is 0.087. These are run-to-run numerical differences under a newer CPU PyTorch build; the eight-window interpolation result will be calculated from the rebuilt models themselves.
14. At about 20:21 IST, windows 5 and 6 completed: **30/40** arrays are present. The last two windows are training. No final interpolation result has been calculated yet.
15. At about 20:34 IST, window 7 completed: **35/40** arrays are present. Window 8 remains in progress. No final interpolation score has been calculated yet.
16. At about 20:35 IST, window 8 completed and the training command exited **0**. `summary.json` contains **40/40 finite runs**, eight windows (1–8) and all five seeds (0–4); 40 trajectory and 40 residual arrays are present. Training log: `results/interpolation_rebuild/pinn_train.log`.
17. First full experiment-09 execution used fine-grid prediction times. It produced `results/interpolation_rebuild/interpolation/metrics.json`, but this version allowed linear interpolation at times between sparse knots to use a residual sample from *after* the prediction time. **Do not use those predictive scores**: this is a look-ahead design error. The run is retained as an audit trail. Reconstruction fidelity from it remains a valid numerical comparison.
18. Corrected experiment 09 to evaluate prediction **only at sampled knots** (spacing 0.05). Every residual knot used to construct a trailing feature is then at or before prediction time. Reran all 40 runs into `results/interpolation_rebuild/interpolation_sampled_knots/metrics.json`; exit **0**. Added paired linear-versus-hold comparisons and per-window squared-error reductions, then reran the same corrected analysis to update that metrics file. All models at a given horizon use identical samples.
19. Validation after correction: **40 finite runs**, 40 trajectory arrays, 40 residual arrays; `tests/test_interpolation.py` plus `tests/test_analysis.py`: **11 passed**; Python compilation and `git diff --check` passed.

### Final exploratory results from the corrected 40-run analysis

Linear interpolation reconstructs the dense trailing-average magnitude more accurately than sample-and-hold: mean relative absolute error across runs is **0.079 vs 0.108** at width 0.1, and **0.071 vs 0.079** at width 0.25. This measures recovery of the residual signal, independently of predictive value.

The table shows the gain in held-out `R²` over a baseline with pointwise log residual magnitude and normalized time. Parentheses contain the 95% bootstrap interval over **eight initial-condition windows**. The last column is the direct linear-versus-hold gain, followed by the number of windows (of eight) in which linear reduces squared error relative to hold.

| Future horizon | Trailing width | Baseline `R²` | Linear gain | Hold gain | Linear − hold gain; positive windows |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.10 | 0.197 | +0.288 (+0.090, +0.526) | +0.186 | +0.102 (+0.001, +0.200); 5/8 |
| 0.05 | 0.25 | 0.197 | +0.474 (+0.308, +0.946) | +0.416 | +0.058 (+0.003, +0.111); 4/8 |
| 0.10 | 0.10 | 0.279 | +0.295 (+0.171, +0.537) | +0.207 | +0.088 (+0.016, +0.170); 5/8 |
| 0.10 | 0.25 | 0.279 | +0.472 (+0.341, +1.010) | +0.417 | +0.055 (+0.021, +0.100); 6/8 |
| 0.25 | 0.10 | 0.568 | +0.216 (+0.133, +0.757) | +0.162 | +0.054 (+0.037, +0.124); 8/8 |
| 0.25 | 0.25 | 0.568 | +0.312 (+0.161, +1.153) | +0.260 | +0.052 (+0.032, +0.118); 7/8 |
| 0.50 | 0.10 | 0.817 | +0.049 (−0.006, +0.231) | +0.002 | +0.047 (+0.029, +0.188); 7/8 |
| 0.50 | 0.25 | 0.817 | +0.043 (−0.047, +0.257) | −0.002 | +0.046 (+0.025, +0.190); 7/8 |

**Conclusion for this follow-up:** A causal-at-evaluation-time trailing residual average carries substantial held-out information about future *total* error over horizons 0.05–0.25 in these rebuilt Lorenz PINNs. Linear interpolation adds a smaller pooled gain over sample-and-hold. At horizon 0.5, the linear average does not reliably improve on the pointwise-and-time baseline. The positive paired linear-versus-hold gain at 0.5 does not change that limitation. This is an exploratory result, with two widths and four horizons examined and only eight independent initial-condition windows. It does not overturn the original negative conclusion about **pointwise residual direction**, and it does not test a Duhamel-weighted predictor or improve PINN training.

The original 40-run directional study and this follow-up use different prediction-time subsets, a different baseline (time is included here), and models rebuilt with a newer CPU PyTorch version. Their `R²` values should not be compared directly. The dense-residual trailing average sometimes predicts worse than the sparse linear reconstruction; smoothing could be one explanation, but that mechanism was not isolated experimentally.

### Current result

There is **no empirical Lorenz interpolation result yet**. The original negative conclusion remains the only tested project conclusion. The next work is to recover or regenerate reference and PINN arrays, then run experiment 09. If regeneration occurs, use a separate result directory and document the commands, settings, run counts, outputs, and comparison against the baseline here.
