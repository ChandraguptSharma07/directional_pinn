# Independent validation run log — 22 November 2026

The presentation-facing validation follows
[`independent_validation_plan.md`](independent_validation_plan.md), frozen
before seeing the new windows' outcomes. This log records each completed
stage and any deviation. The original 40-run results remain separate.

## Start

The fresh reference configuration changes only the long-run seed state to
`[7, 8, 9]`. The unchanged model will be evaluated on windows 1–8,
seeds 0–4. Outputs go under `results/independent_validation/`.

## Reference run

`experiments/03_lorenz_reference.py --config
config/independent_validation.yaml --out
results/independent_validation/reference` completed with **36/36 checks
passed**. The minimum reference trust horizon across the nine windows is
5.83, exceeding the model's two-unit horizon. Window 0 remains unused.
All eight evaluation initial states differ from the prior reference set;
model settings remain frozen.

## Tangent-direction run

`experiments/04_lorenz_lyapunov.py` with the frozen configuration and new
reference completed with **31/31 checks passed**. Its 180-unit Lyapunov
exponents were approximately `[0.9015, -0.0023, -14.5658]`; their sum
matched the Lorenz Jacobian trace. The new window-specific tangent arrays
are in `results/independent_validation/lyapunov/data/` and are available
for the directional test. The 40 PINN trainings are still underway.

## Code and configuration check

The full test suite passed **129/129** while the validation trained;
`git diff --check` was clean. A direct config diff confirmed that the
fresh validation config differs from the default only in the long-run
seed state. No training or analysis setting was changed.

## PINN batch in progress

The frozen 8-window × 5-seed batch is training under
`results/independent_validation/pinn/`. Early completed runs already show
late failures on all five seeds of new window 1 and on some seeds of new
window 3, while completed window-2 runs stay below error 1. These are
provisional observations, not the aggregate result. No settings are being
changed and no runs are being removed. The final summary will be recorded
after all 40 complete.

The batch survived the interrupted assistant turn. At the resume check,
**20/40** run metrics were saved and the training process was still active.
The frozen configuration and outputs were left intact.

## Frozen 40-run PINN validation complete

All **40/40** runs finished under the fixed model settings. The aggregate
record is `results/independent_validation/pinn/validation_40runs_summary.json`.
The median of run-level median errors is **0.0551** and the median of
run-level maximum errors is **0.3991**. **30/40** runs stay below error 1
throughout the two-unit interval; **32/40** stay below it for at least
90% of the interval. This is below the development set's 35/40
throughout result, so the earlier accuracy claim does not transfer at
the same rate. Every run, including failures, remains saved.

Per-window counts of runs below error 1 throughout are: window 1: 0/5;
2: 5/5; 3: 2/5; 4: 5/5; 5: 5/5; 6: 4/5; 7: 4/5; 8: 5/5. The model is
therefore usually accurate, but difficult initial conditions can affect
multiple seeds together. The figure
`docs/figures/independent_validation_40runs.pdf` was generated and visually
inspected. The 40 runs were packaged for the existing direction analysis
without altering their arrays. The original decision-rule analysis and a
separate whole-window bootstrap sensitivity are running.

## Independent directional test complete

The unchanged Phase 7 analysis completed on **40 finite new runs** with
31,360 usable predictor samples after exact segment joins were excluded.
Neither `b1` nor `s1` alignment passed any of the original five horizons;
the original four-part rule again returned **not supported** for both.
The held-out gains in R² from `b1` ranged from −0.0043 to +0.0062,
and from `s1` from −0.0079 to −0.0003. Restricting to samples with current
and future trajectory error below 1 still gave near-zero or negative gains
for both directions at every horizon (29,933 samples). Full output:
`results/independent_validation/directional_analysis/`.

The full-propagator oracle feature, which uses the known reference,
increased held-out R² beyond residual size by +0.0664, +0.2672, and
+0.3691 at future horizons 0.25, 0.5, and 1.0. The separate analysis
with elapsed time and its square in the baseline, and **whole-window
bootstrap clusters**, found gains of +0.056 [0.031, 0.083], +0.258
[0.206, 0.395], and +0.442 [0.267, 0.672] at those horizons.
Its full output is `results/independent_validation/window_analysis/`.
These intervals describe uncertainty across the eight windows; the
oracle is not a deployable predictor.

This independent set strengthens a narrow conclusion: pointwise residual
alignment with one expanding direction does not improve held-out future-
error prediction for the tested Lorenz PINNs, including when the model
usually follows the reference. It also makes the model-quality claim more
honest: the sequential model is markedly better than the global T=2
pilot, but only 30/40 new runs stayed below error 1 throughout.

## Presentation artifacts

The two-page professor briefing was updated with the **independent**
30/40 result and the repeated negative direction finding; it is in
`docs/professor_project_brief_2026-11-22.pdf` with Markdown and LaTeX
sources. The earlier three-page research update now carries a visible
independent-validation addendum so its developmental 35/40 figure is not
mistaken for the new result. `docs/presentation_2026-11-23.pdf` is a
four-slide presentation built directly from the frozen metrics. All four
slide previews were visually inspected. The figure for the fresh 40 runs
is `docs/figures/independent_validation_40runs.pdf`.

## Final verification before presentation

The final full test suite passed **129/129 in 10.24 seconds** and
`git diff --check` was clean. The four-slide PDF, speaking notes, updated
professor brief, fresh 40-run figure, accuracy summary, original-rule
analysis, and whole-window uncertainty metrics were confirmed present.
The professor and research-update PDFs compiled without undefined
references or overfull boxes. No model setting was changed after the
independent windows were generated, and no validation run was omitted.

**Presentation conclusion:** The initial pointwise one-vector residual
hypothesis is not supported for the tested Lorenz PINNs. This result held
in the original planned test and in the frozen fresh-window follow-up.
The sequential model extends useful trajectory length but has
initial-condition-dependent failures: 30/40 fresh runs are below error 1
throughout T=2. Full oracle amplification adds held-out information at
longer horizons, with whole-window uncertainty intervals above zero, but
requires the known reference and is not a practical predictor.
