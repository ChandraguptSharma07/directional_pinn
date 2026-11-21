# Model improvement: return to the original direction question

## 2026-11-21 session

The current objective is to improve Lorenz PINN trajectory accuracy enough to
revisit the original question: whether residual alignment with the leading
instability direction predicts future error beyond residual magnitude. The
method comparison and interpolation study are exploratory background, not the
main claim. The fixed, earlier 8-window direction test was negative; its result
must remain visible even if a new model produces a different outcome.

The Phase 4 pilot shows a specific obstacle: with the original soft initial
condition penalty, the T=2, window-0, seed-0 run had median trajectory error
17.476, max error 40.649, and error below 1 for only 3.0% of the window,
despite median residual 0.083. This is a failure of the trained trajectory,
not evidence for the direction hypothesis. Source:
`results/pinn_pilot/runs/T2_lambda100/metrics.json`.

The first focused change is an optional hard initial condition. For raw network
output `g(t)`, use `x_hat(t) = x0 + (t/T) [g(t)-x0]`. This guarantees
`x_hat(0)=x0` exactly. It is enabled by `TrainConfig.hard_ic`; the original
configuration retains its previous behavior. The stacked implementation uses
each member's own initial state. `experiments/11_hard_ic_pilot.py` runs the
same T=2, window-0, seed-0 setup with 20,000 iterations. We will judge it by
trajectory error against the trusted reference, not by residual alone.

Verification and pilot results will be appended below after each run.

### Verification run 1

The pre-change full suite completed with **124 passed**. The first targeted
post-change run had **16 passed, 1 failed**: unstacking a batched model copied
the first member's initial state into all members. This exposed a real issue
for training multiple windows together. `StackedPINN.unstack` now copies the
selected member's initial state; the targeted suite is being rerun.

### Verification run 2

After the fix, the targeted PINN suite passed: **17 passed**. The hard
initial-state model gives the exact specified state at `t=0`, its time
derivative agrees with finite differences, and the stacked evaluation agrees
with individual models for different initial conditions. The 20,000-step
trajectory pilot is next.

### Verification run 3

The full post-change suite passed: **126 passed in 9.91 seconds**. This
confirms the original tests still pass with the optional model change. It
does not establish trajectory improvement; that requires the pilot result.

### Trajectory run 1: hard initial condition, T=2

Command: `python experiments/11_hard_ic_pilot.py` using the repository virtual
environment and required Nix library path. Window 0, seed 0, 20,000 Adam
steps, 1,024 collocation points, float32, the existing time curriculum, and
all baseline settings except `hard_ic=True`. The run took 128.8 seconds.
Output: `results/hard_ic_pilot/w0_t2_s0/` (configuration, model, history,
trajectory, residuals, and metrics).

| Metric | Original soft IC pilot | Hard IC pilot |
|---|---:|---:|
| Initial error | 0.0772 | 0.000000339 |
| Median trajectory error | 17.476 | 6.052 |
| Maximum trajectory error | 40.649 | 35.767 |
| Fraction with error < 1 | 3.0% | 11.3% |
| Median residual norm | 0.083 | 0.819 |

The hard IC improves the trajectory but remains inadequate for a two-unit
direction study: error first exceeds 1 at `t=0.226`. At `t=1.25`, trajectory
error is 32.1 while the local residual norm is 0.11. Early residuals are much
larger (`2.36` at `t=0`, `3.15` at `t=0.1`), and their errors can propagate
forward. A low late residual does not certify the entire trajectory.

The baseline metrics come from the earlier pilot with matching initial state
and settings, but its original reference trajectory file is unavailable in
this checkout. Treat the side-by-side values as an exploratory comparison,
not a paired rerun under the current environment. The new hard-IC result is
evaluated against the regenerated high-accuracy reference.

## 2026-11-22 continuation: emphasize early residuals

The T=2 hard-IC trajectory begins to diverge at `t≈0.226`; the residual is
largest before this point and small after the model has drifted. The next
exploratory training run uses the same hard-IC architecture and Adam schedule,
but weights the squared ODE residual by `w(t)=1+9(1-t/T)`, normalized by the
sum of weights over collocation points. Early residuals then receive 10 times
the weight of residuals at the end. `early_weight=0` reproduces the original
uniform residual objective. We will test whether this improves trajectory
accuracy, especially the first time error exceeds 1 and the fraction of the
window below that threshold. The choice of 9 is a pilot choice informed by
the observed failure profile; no confirmatory claim will be drawn from it.

### Verification run 4

The PINN tests passed **18/18**, including the exact initial condition and the
weighted residual calculation. The full suite then passed **127/127**. A
further stacked-versus-separate training check was added afterward and still
needs to be run. The weighted 20,000-step pilot is in progress.

### Verification run 5

The expanded targeted suite passed **19/19**. It includes a short training
check that stacked hard-IC, early-weighted runs with different initial states
match the corresponding individually trained runs to floating-point tolerance.

### Trajectory run 2: hard IC plus early weighting, T=2

Command: `python experiments/11_hard_ic_pilot.py --early-weight 9 --out
results/hard_ic_early_weight_pilot/w0_t2_s0` with the same environment as run
1. Window 0, seed 0, 20,000 Adam steps, 1,024 points, float32. Runtime: 131.5
seconds. Output directory contains the full run record.

Median trajectory error was **6.036**, max error **35.817**, and error was
below 1 for **12.6%** of the window. The median residual norm was **1.119**;
final error was **12.398**. Relative to hard IC alone (median error 6.052,
11.3% below 1), this is a very small gain and does not yield a usable T=2
trajectory. The stronger early weight is not supported as the main model
improvement from this pilot. We will retain it as an optional exploratory
setting but keep the default at zero.

### Planned optimizer refinement

The next run starts from the saved unweighted hard-IC Adam model and applies
300 L-BFGS steps to the full T=2 window in float64. It uses 1,025 fixed
collocation times, the same unweighted ODE residual loss, and no reference
states during training. The reference is only used for evaluation. This tests
whether the Adam run is limited by incomplete optimization. The new result
will be saved separately in `results/hard_ic_lbfgs_pilot/w0_t2_s0/`.

### Trajectory run 3: L-BFGS refinement

`python experiments/12_lbfgs_refine.py` ran 300 L-BFGS iterations (310 loss
evaluations) in float64, starting from the saved 20,000-step hard-IC model.
Runtime was 5.19 seconds. Full-window collocation loss moved from **2.31833**
to **2.31595**, while median trajectory error stayed at **6.052** and the
fraction with error below 1 stayed at **11.3%**. The optimizer found no
meaningful improvement in this basin. This does not prove that the global
residual optimum is poor, but a simple local refinement is insufficient.
Full run record: `results/hard_ic_lbfgs_pilot/w0_t2_s0/`.

### Planned piecewise pilot

The global T=2 network appears trapped near an incorrect easy trajectory.
We will train eight independent hard-IC networks on consecutive intervals of
length 0.25, with 4,000 Adam steps and 256 collocation points each. The next
segment takes the previous network's predicted endpoint as its initial state;
it does not take the true reference state there. Thus the method remains a
PINN initial-value solve, and errors are free to accumulate. Reference states
are only used for after-training evaluation. The test is whether this local
representation retains useful accuracy over a two-unit trajectory.

### Trajectory run 4: piecewise hard-IC PINN, T=2

`python experiments/13_piecewise_pinn_pilot.py` trained eight 0.25-long
segments on window 0, seed 0, without intermediate reference anchors.
Runtime was **90.63 seconds** for 32,000 total Adam steps. The output in
`results/piecewise_pinn_pilot/w0_t2_s0/` includes each segment's weights and
history plus stitched trajectories, residuals, metrics, and a run record.

| Metric | Single hard-IC PINN | Piecewise hard-IC PINN |
|---|---:|---:|
| Median trajectory error | 6.052 | **0.0259** |
| Maximum trajectory error | 35.767 | **0.1540** |
| Fraction with error < 1 | 11.3% | **100%** |
| Final trajectory error | 6.728 | **0.1540** |
| Median residual norm | 0.819 | 0.398 |

This is a substantial pilot improvement: a two-unit PINN trajectory is now
accurate on this initial condition. The piecewise model has eight networks,
32,000 total updates, and 256 collocation points per update; the global model
has one network, 20,000 updates, and 1,024 points per update. The result
therefore supports this *package* of local representation and training, not an
isolated causal effect of segmentation. It is still a single pilot initial
condition. We need locked-setting runs on unseen windows before using it to
revisit the directional hypothesis.

The two setups also use different total collocation evaluations: about
20.48 million for the single-network Adam run and 8.19 million for the
piecewise run. On this machine, measured training times were 128.8 and
90.6 seconds respectively. These are implementation-specific costs, not a
controlled efficiency benchmark.

### Locked validation set before unseen runs

For the next accuracy check, we fix exactly the pilot settings above: eight
segments of length 0.25, 4,000 Adam updates per segment, 256 stratified
collocation points, hard IC, no early weighting, no curriculum within a
segment, and seed 0. We will run windows 1–8 once each. These windows are
unseen by this model choice, but the choice was motivated by the earlier
project's analysis; the result is an exploratory validation, not a fresh
preregistered test. We will report all eight, including failures, and will
not retune on them before summarizing this check.

The pilot figure is in `docs/figures/piecewise_pinn_t2_pilot.pdf` and `.png`.
It shows all three state components against the trusted reference and the
trajectory error on a logarithmic scale. It was generated by
`experiments/14_piecewise_figure.py`; the plot is descriptive of window 0
only.

The saved segment models were reloaded and checked at all seven joins using
`experiments/15_piecewise_diagnostics.py`. The largest state jump is
`9.99×10⁻⁷`, consistent with transferring endpoints through float32 initial
states. This is far below the observed trajectory error; the stitched plot
does not hide a large discontinuity.

The same reload check found derivative jumps as large as **4.00** at joins.
The derivative, and hence the pointwise ODE residual, has distinct left and
right values at those exact times. The original analysis samples exact join
times because 0.25 is on its 0.01 statistics grid. We added an opt-in
`--exclude-joins 0.25` setting that drops only those predictor times for
piecewise runs; the original smooth-PINN analysis remains unchanged. The
standard Duhamel trapezoid diagnostic does not account for two-sided slopes,
so its reported reconstruction mismatch for piecewise runs must not be used
as evidence about the linear error mechanism without a corrected quadrature.

The original directional analysis also requires the reference-based tangent
directions. Their raw arrays are absent in this checkout, so we are
regenerating them with the existing Phase 3 experiment from the regenerated
reference. This is preparation for a later direction test on the new PINN
outputs; no directional conclusion is being claimed from the pilot alone.

### Verification run 6

After adding the piecewise experiment and diagnostics scripts, the full
test suite passed **128/128 in 14.95 seconds**, and `git diff --check`
reported no whitespace errors. The window-0 figure was also visually
inspected after generation.

### Trajectory runs 5–12: locked unseen-window check

The fixed piecewise setup was run once at seed 0 on every analysis window
1–8. No settings were changed between these runs. Each output directory is
`results/piecewise_pinn_pilot/w<window>_t2_s0/` and contains the segment
models, histories, stitched trajectory, residuals, metrics, and run record.

| Window | Median error | Maximum error | Fraction with error < 1 | First time error > 1 |
|---:|---:|---:|---:|---:|
| 1 | 0.0490 | 8.2064 | 83.6% | 1.674 |
| 2 | 0.0597 | 0.4658 | 100% | never |
| 3 | 0.0471 | 0.2103 | 100% | never |
| 4 | 0.0305 | 0.1164 | 100% | never |
| 5 | 0.0299 | 0.1380 | 100% | never |
| 6 | 0.0432 | 0.2615 | 100% | never |
| 7 | 0.0250 | 0.0621 | 100% | never |
| 8 | 0.0436 | 0.1574 | 100% | never |

Across these eight, the median of per-window median errors is **0.0434** and
the median of maxima is **0.184**. Seven of eight remain below error 1
throughout. Window 1 is a real failure at late times: its maximum error is
8.21. This shows the model can make two-unit trajectories accurate for most
tested initial states with the fixed settings, but it is not uniformly
reliable. The result is seed-0 only and does not yet support a directional
prediction claim.

### Reference-direction regeneration

`experiments/04_lorenz_lyapunov.py --reference
results/interpolation_rebuild/reference --out
results/interpolation_rebuild/lyapunov` completed with **31/31 checks passed**.
It regenerated the tangent propagators and direction arrays for all nine
reference windows. The estimated Lyapunov exponents over 180 time units were
`[0.9237, -0.0249, -14.5655]`, whose sum agrees with the Lorenz Jacobian
trace to the reported precision. These arrays are ready for an exploratory
direction analysis of the new piecewise PINNs.

### Exploratory direction analysis plan

The original Phase 7 analysis expects one run directory per window and seed,
with a `finite` flag. `experiments/16_prepare_piecewise_analysis.py` will
package the eight seed-0 piecewise outputs in that format without changing
their arrays. We will then run the existing direction-analysis code with the
regenerated Lyapunov data. This is a *one-seed exploratory analysis* of a
model selected after the original negative test, so its intervals and verdict
must not be presented as a fresh confirmatory answer. It can show whether the
better trajectory changes the diagnostic pattern and motivate a later fixed
five-seed rerun.

### Locked five-seed extension

Without changing the eight-segment settings, `experiments/17_piecewise_batch.py`
will train seeds 1–4 for every window 1–8. Together with the completed seed-0
runs, this gives the same 8 × 5 window/seed structure as the original study,
now over T=2 trajectories. Each run is kept regardless of quality. The batch
summary will report the fraction of runs that remain below error 1 throughout
and for at least 90% of the window. This is a robustness check for the new
model, followed by a post-hoc rerun of the direction analysis. It does not
retroactively change the original prespecified negative result.

### Directional analysis run 1: eight windows, seed 0

The existing `experiments/07_directional_analysis.py` was run on the eight
piecewise seed-0 outputs with the regenerated Lyapunov arrays. It completed
with 8 finite runs and 6,520 samples; output is in
`results/piecewise_seed0_directional_analysis/`. The residual-size-only
baseline had negative held-out R² at every horizon (−0.015 to −0.132). The
alignment with `b1` never helped. Alignment with `s1` had a small held-out
gain at horizon 0.25 (+0.0159, reported 95% interval +0.0048 to +0.0326),
below the fixed +0.02 threshold; at 0.5 its +0.0225 gain had a wide interval
that crossed zero. Neither direction passed the full original decision rule
at any horizon. This **does not establish a new negative result**: it uses
only one seed per window, a post-hoc model design, and residuals from a
piecewise function. The five-seed extension is in progress.

### Directional analysis run 2: exact joins excluded

After adding `--exclude-joins 0.25`, the seed-0 analysis used 6,272 samples
(248 fewer than the first run). The held-out pattern barely changed: at
horizon 0.25, the `s1` alignment gain moved from +0.0159 to **+0.0119**;
`b1` remained negative. Neither direction passed the original decision rule
at any horizon. Output is in
`results/piecewise_seed0_directional_analysis_no_joins/`. This sensitivity
check reduces ambiguity from the two-sided residual at exact segment joins;
it still has only eight runs. The reported Duhamel reconstruction diagnostic
in that output remains uncorrected for slope jumps and should be ignored.

### Trajectory runs 13–44: five-seed extension complete

`experiments/17_piecewise_batch.py` completed all 32 remaining runs without
changing the model settings. Together with seed 0, there are **40/40 finite
saved trajectories** in `results/piecewise_pinn_pilot/`. The aggregate record
is `validation_40runs_summary.json`. The median across run-level median
trajectory errors is **0.0433**; the median of run-level maximum errors is
**0.1823**. **35/40 runs** remain below error 1 throughout the two-unit
window. The same 35/40 satisfy the earlier pilot's 90%-of-window criterion.
All five failures are window 1, where error rises above 1 late in the
trajectory. Their maxima range from about 3.05 to 8.21. Every run on windows
2–8 stays below 1 throughout. The distribution is shown in
`docs/figures/piecewise_pinn_validation_40runs.pdf` and `.png`, generated by
`experiments/18_piecewise_validation_figure.py` and visually inspected.

This is a substantial model-quality advance over the length-one limit of the
original global PINN, though it does not guarantee a reliable length-two
solution on all initial conditions. Because the model was developed after
seeing the original negative direction result, the direction rerun on these
40 runs is exploratory even though these exact settings were held fixed
throughout the five-seed extension.

The 40 trajectories were packaged without altering their arrays in
`results/piecewise_40runs_analysis_runs/` using
`experiments/16_prepare_piecewise_analysis.py`. The original Phase 7 analysis
is running with `--exclude-joins 0.25` and the regenerated tangent data.

### Directional analysis run 3: all 40 piecewise PINNs

The Phase 7 analysis finished with 40 finite runs and **31,360 usable
samples** after excluding exact joins. Full output is in
`results/piecewise_40runs_directional_analysis_no_joins/`.

| Horizon | R², residual only | ΔR² with b1 alignment | ΔR² with s1 alignment | ΔR² with oracle full amplification |
|---:|---:|---:|---:|---:|
| 0.05 | −0.017 | −0.0003 | −0.0041 | −0.0005 |
| 0.10 | −0.019 | −0.0001 | −0.0020 | +0.0027 |
| 0.25 | +0.009 | −0.0099 | +0.0029 | +0.0513 |
| 0.50 | −0.033 | −0.0176 | +0.0010 | +0.1948 |
| 1.00 | −0.067 | −0.0002 | −0.0013 | +0.3068 |

Neither one-vector alignment passes any horizon of the original four-part
decision rule. For the oracle `||Φ(t+Δ,t) r(t)||`, which uses the reference
trajectory and all directions, the reported run-bootstrap intervals for
the gains at horizons 0.25, 0.5 and 1.0 are respectively **[+0.018,
+0.078]**, **[+0.118, +0.248]**, and **[+0.148, +0.414]**. This is a
stronger association than in the original length-one PINNs, but it is
secondary and oracle-only. The one-vector negative result is consistent
with the original finding on a more accurate and longer set of models, yet
the new model and analysis were developed after seeing the original result.
It is supportive exploratory evidence, not a second preregistered test.

The residual-only held-out R² is near zero or negative at most horizons.
Because the target is *total future error*, which includes already
accumulated error, that baseline is weak. Before highlighting the oracle
gain as a mechanistic result, we will test whether it adds beyond elapsed
time, a simple possible confound. The piecewise Duhamel reconstruction
diagnostic in this Phase 7 output remains uncorrected for derivative jumps.

### Post-hoc check of the oracle signal

`experiments/19_piecewise_oracle_controls.py` will add elapsed time and its
square to the residual-size baseline, then test whether the oracle full
amplification still helps on held-out windows. It will also show a sensitivity
without the late-failing window 1, and a more stringent oracle baseline that
includes the current true error. These are new diagnostic models chosen after
seeing the results; they cannot be treated as part of the original fixed
decision rule. The current-error baseline itself requires the reference and
is only for mechanism analysis.

### Verification run 7

After adding join exclusion and the batch analysis tools, the full suite
passed **129/129 in 10.11 seconds**; `git diff --check` was clean. The
40-run validation figure was generated and visually inspected.

### Oracle control run: elapsed time and current error

`experiments/19_piecewise_oracle_controls.py` completed and saved full
held-out results in `results/piecewise_40runs_oracle_controls/metrics.json`.
The baseline now includes `log ||r||`, time `t`, and `t²`; prediction is still
leave-one-window-out, with run-cluster bootstrap intervals. Adding the oracle
`log(||Φr||/||r||)` gives these changes in held-out R²:

| Horizon | All eight windows | Without window 1 |
|---:|---:|---:|
| 0.05 | −0.002 | −0.002 |
| 0.10 | +0.004 | −0.001 |
| 0.25 | **+0.046** [0.024, 0.064] | **+0.026** [0.008, 0.042] |
| 0.50 | **+0.191** [0.132, 0.233] | **+0.135** [0.076, 0.180] |
| 1.00 | **+0.360** [0.213, 0.469] | **+0.169** [0.059, 0.286] |

The time terms alone markedly improve the short-horizon baseline (R² 0.407
at horizon 0.05, versus −0.017 for residual size alone). The oracle gain at
longer horizons persists after controlling for time and after removing the
single difficult window. In a further mechanistic check, including the
*current true error* in that baseline still leaves oracle gains of +0.078,
+0.203, and +0.361 at horizons 0.25, 0.5, and 1.0 across all windows;
the gains are +0.051, +0.150, and +0.175 without window 1. In the same
time-controlled models, adding `s1` alignment gives negligible gain.

These are post-hoc associations, not causal proof or a deployable error
estimator: both `Φ` and current true error require the reference trajectory.
The bootstrap resamples runs that share windows, so the intervals should be
interpreted as descriptive uncertainty under the implemented procedure,
not independent replication across 40 initial conditions.

### Research update and final verification

The concise, shareable report is
`docs/piecewise_pinn_research_update.pdf` (three pages), with editable
LaTeX and Markdown sources of the same name. It includes the pilot and
40-run figures, the original question, the model method, the negative
one-vector result, the positive oracle diagnostic, and the major limits.
LaTeX compiled twice without undefined references or overfull boxes.
The final full suite passed **129/129 in 9.96 seconds**; `git diff --check`
was clean. The two figure PDFs, research-update PDF, 40-run accuracy summary,
40-run direction metrics, and time-control metrics were confirmed present.

**Current conclusion:** the piecewise hard-IC model is a substantial
trajectory-quality improvement for Lorenz windows of length two, but window
1 fails late for every seed. A better trajectory does not rescue the tested
single-direction residual alignments. Full oracle amplification has a
positive held-out association at longer horizons after simple time and
window-1 controls, but it is not a practical predictor and this follow-up
is post-hoc. New untouched windows and a causal residual-history predictor
are the next tests needed for a stronger paper claim.

### Terminology clarification

The descriptive name for the implemented construction is **sequential
time-domain decomposition PINN** (also **time-marching PINN**): independent
networks are trained on successive short intervals, and each receives the
previous model's predicted endpoint as its starting state. These terms are
used in the temporal-decomposition literature, including Penwarden et al.
([arXiv:2302.14227](https://arxiv.org/abs/2302.14227)). Our hard starting
state resembles the idea in Roy and Castonguay's hard-constrained sequential
PINN ([arXiv:2403.03223](https://arxiv.org/abs/2403.03223)), but our saved
model has float32 state jumps near `10⁻⁶` and derivative jumps at joins. We
therefore use the generic descriptive name, not their specific acronym.

### Supervisor briefing prepared

`docs/professor_project_brief_2026-11-22.md` and its two-page PDF now tell
the complete project story in third person: original goal and hypothesis,
prespecified negative result, plausible explanations kept separate from
observations, the sequential-model improvement, the exploratory oracle
finding, and the evidence required for a stronger conclusion. The PDF
compiled without LaTeX warnings or overfull boxes, and `git diff --check`
was clean. This documentation run did not change experiment data or the
original decision rule.
