# Final LaTeX report assembly — 22 November 2026

This running record documents the report-writing instance and its evidence. It supplements, rather than replaces, the experimental logs.

## Scope and source check

- Requested a complete LaTeX report that starts with the original hypothesis and reference paper, explains the loss functions and mathematical mechanism, follows the tested failures and continuation, and ends with the freshest conclusion.
- Checked the official arXiv HTML of Turinici, *Regime-Aware Time Weighting for Physics-Informed Neural Networks*, v2, especially its residual loss (Eq. 6), one-sided error bound (Eqs. 8–9), minimax time density (Eqs. 12–15), and practical growth proxy (Eq. 17). The report distinguishes the paper's adaptive weight from our fixed linear early-time pilot.
- Reconciled values and design status against `docs/analysis_plan.md`, `docs/results.md`, `docs/interpolation_study.md`, `docs/interpolation_progress.md`, `docs/model_improvement_progress.md`, `docs/phase0_linear_saddle.md`, `docs/phase2_lorenz_reference.md`, `docs/phase3_lyapunov.md`, `docs/phase4_pinn.md`, and `docs/validation_progress_2026-11-22.md`.
- The original submitted PDF and reference PDF were present locally. No PDF text extractor was installed, so the submitted report's detailed phrasing was not quoted; the full project record and official paper text supplied the report's substantive evidence.

## Artifact and conclusion

- Wrote `docs/final_project_report.tex`, covering the original question, paper derivation, Lorenz equations/Jacobian, exact and linearized error dynamics, toy saddle, two tangent directions, original PINN and its loss, fixed held-out test, original negative result, interpolation with corrected causal evaluation, hard IC and early-weight/L-BFGS pilots, sequential segment equations, and all 40 fresh-window validation runs.
- Included the existing original analysis, pilot, and fresh validation figures. The report's final statement is bounded: one-vector pointwise alignment does not meet the original predictive rule in either set; sequential training usually improves the two-unit trajectory but has 10/40 fresh full-window failures; full-propagator oracle and residual history are informative only under their stated exploratory/reference-dependent conditions.
- Preserved the distinction between the original prespecified negative result, subsequent exploratory method development, and the frozen fresh-window validation. Untested ideas are identified as such.

## Build verification

- Built with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp docs/final_project_report.tex` from the repository root. The final PDF is `docs/final_project_report.pdf`, 12 pages. LaTeX resolved references and citations. A layout pass removed all material overfull boxes; one 1.65 pt overflow in the prose remains, with no clipped table or figure reported. `git diff --check` was clean. The final PDF and source are both in `docs/`.

## Later continuation: reference-paper adaptive weighting

The user asked to implement and test Turinici's adaptive time weighting. The official author's repository was cloned and audited; its exact commit, time-step discrepancy, implementation, every training run, and matched control results are recorded in `docs/adaptive_weighting_progress_2026-11-22.md`. The LaTeX report gained a full method/results section with the paper-derived rate and loss equations, a global pilot, the eight-window paired sequential table, and a separate literal source-code time-scale sensitivity table. The conclusion now states that neither scaling improved the solver in these tests. The original directional result and fresh 30/40 model validation remain unchanged.

The updated PDF built successfully as **14 pages**, with resolved citations and references and no material box overflow. It is `docs/final_project_report.pdf`; its source is `docs/final_project_report.tex`. The full suite passed **131/131 in 10.06 seconds** after both weighting variants were implemented; `git diff --check` was clean. The equation-based/uniform summary has **16/16** rows and the literal-code summary **8/8** rows across all eight windows, with no missing per-run metric file. The final PDF was copied into `docs/` after the last successful build.

## Report expansion requested later on 22 November

The user asked for a more detailed report of about 15–16 pages with evidence from the run logs. No new numerical model training was necessary. I rechecked values against saved pilot metrics, the interpolation progress log, and the 40-run frozen validation summary. The LaTeX source now includes:

- a table of held-out interpolation gains for linear reconstruction, sample-and-hold, and the sampled-knot box control at width 0.25, plus the width-0.10 sensitivity in prose;
- a five-row run-log table tracing the window-0 `T=2` path from soft IC through hard IC, early weighting, L-BFGS, and eight-segment training, with trajectory error, threshold share, and median residual;
- the concrete hard-IC failure example at `t=1.25`, where trajectory error was about 32 while the instantaneous residual was about 0.11;
- a fresh-validation table for all eight starting-condition windows, each showing the median of five run medians, worst maximum error, and full-window success count. It explicitly explains why low median error can hide late failure and why eight windows, not 40 seeded runs, are the independent initial-condition units.

The final `latexmk` build produced **15 pages** with resolved references and no material box overflow. The updated PDF was copied to `docs/final_project_report.pdf`. These additions are descriptive evidence from prior runs and do not change the original directional verdict or promote exploratory interpolation to a confirmatory result.

Final verification: `git diff --check` exited cleanly; the LaTeX log shows no undefined reference or citation. The only remaining layout notice is a 1.65 pt prose overflow near the reference-paper title, with no table or figure overflow. The PDF and source are both present in `docs/`.

## Additional figures requested later on 22 November

The user asked for more figures. Added `experiments/23_report_figures.py`, which reads saved results rather than hardcoding plotted values and creates vector PDF plus PNG previews under `docs/figures/`:

- `report_interpolation_gains`: held-out $R^2$ gain versus future horizon for linear interpolation, sample-and-hold, and the sampled-knot average at trailing width 0.25. The caption distinguishes the larger history effect from the smaller interpolation-specific effect and notes the uncertain 0.5-horizon gain.
- `report_adaptive_paired_windows`: the eight matched seed-0 sequential windows under uniform, paper-equation, and literal author-code time scaling. A log vertical axis separates the close uniform/equation curves from the poorer source-scale outcomes; the caption identifies this as an exploratory comparison within our architecture.

Both PNG previews were visually inspected. The adaptive figure's initial minor log-tick labels and legend overlap were corrected before inclusion. The figures are now in `docs/final_project_report.tex` with captions and reproducibility pointers. `latexmk` produced a **16-page** PDF with resolved references and no figure/table overflow, and the result was copied to `docs/final_project_report.pdf`. No new model experiment was run in this figure-only session.
