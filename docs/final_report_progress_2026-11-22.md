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
