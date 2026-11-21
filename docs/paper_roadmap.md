# Roadmap to a paper-ready report

Written 21 November 2026 after the completed directional study and the exploratory interpolation follow-up. This is a research and writing plan, not a claim that the current evidence is already sufficient for peer-reviewed publication.

## Working thesis

**Question:** For a PINN approximating a chaotic ODE, which residual information is useful for estimating later trajectory error: its instantaneous size, its direction relative to expanding dynamics, or its recent time history?

**Current answer, with scope:** In 40 Lorenz-63 PINNs from eight initial conditions, the original preset test did not support a held-out benefit from pointwise alignment with either the backward Lyapunov vector or the finite-time leading singular vector. A later exploratory analysis on 40 rebuilt PINNs found that a trailing average of residual magnitude improved held-out prediction of future *total* error at short horizons; linear interpolation of sparse residual samples added a smaller gain over simple sampled-knot averaging. This is a **diagnostic** result. No new PINN solver or training improvement was demonstrated.

`experiments/09_residual_interpolation.py` is an exploratory follow-up named by script order. It is **not** the conditional directional-training Phase 9 described in the original README, which was not run after the negative result.

A suitable working title is **“What PINN residuals reveal about future error in chaotic ODEs: direction versus temporal history.”** Do not claim that interpolation is novel or optimal until the literature review and independent validation are complete.

**Provisional case-study abstract:** *We study whether the differential-equation residual of a physics-informed neural network predicts future trajectory error in Lorenz-63. In 40 PINN runs across eight initial conditions, a prespecified held-out test found no reliable gain from projecting a pointwise residual onto a backward Lyapunov vector or a leading finite-time singular vector beyond residual magnitude. An exploratory follow-up on 40 rebuilt runs found that averaging recent residual magnitude improved held-out prediction at short horizons. With sparse residual samples, linear interpolation gave a modest additional gain over a plain average of sampled magnitudes. These findings suggest that temporal residual history can be more informative than pointwise orientation in this setting. The interpolation analysis is exploratory and limited to one chaotic system and one PINN design; independent validation is required before making a general method claim.*

## What the current evidence supports

1. **Preset negative result:** The pointwise expanding-direction hypothesis failed the documented decision rule on Lorenz-63; this is the most defensible statistical result (`docs/analysis_plan.md`, `docs/results.md`).
2. **Exploratory positive result:** On the same eight initial-condition windows, but with rebuilt models and a new analysis subset, recent residual magnitude predicts future total error better than pointwise residual magnitude plus time at horizons 0.05–0.25 (`docs/interpolation_progress.md`). A width-0.25 linearly interpolated average improves held-out `R²` over a plain sampled-knot average by about 0.06 at horizon 0.1 and 0.05 at horizon 0.25. The control was added after inspecting earlier results, so these are **exploratory estimates**.
3. **Methodological lesson:** The first interpolation analysis evaluated between residual sample knots and could use a future knot; its predictive scores are invalid. The corrected analysis evaluates only at observed knots. Keep this correction in the methods and reproducibility record.

The paper should distinguish **residual reconstruction fidelity** from **error-prediction quality**. Better recovery of the dense residual does not by itself show better prediction of trajectory error.

## Main gaps before a strong paper claim

| Gap | Why it matters | Required check |
|---|---|
| Eight independent Lorenz windows | Five seeds per window do not create 40 independent dynamical initial conditions. Bootstrap intervals from eight groups are fragile. | Evaluate a new, untouched set of at least 20–24 separated initial-condition windows, with the same fixed PINN setup. Treat windows as independent units. |
| Post-result choice of interpolation controls and widths | The current positive comparisons were selected and refined while examining results. | Freeze one primary width (0.25), sampling interval (0.05), and two primary horizons (0.1, 0.25) before the new set. Specify `linear` versus `box` as the primary paired comparison; report all runs. |
| Future total error contains inherited error | A trailing residual average could proxy existing error or run difficulty instead of explaining new error production. | Add the created-error target `N_Δ` from the original analysis and a labelled oracle baseline using current true error `E(t)`. Compare against simple cumulative and trailing magnitude controls. Interpret the oracle only as a mechanism check. |
| One chaotic system and one PINN family | Results may reflect this architecture, curriculum, or Lorenz dynamics. | Replicate on a second chaotic ODE with prespecified parameters and at least one simpler linear or stable control. A changed architecture or exact-initial-condition PINN is a further sensitivity test. |
| Sparse-sampling claim | The specific interval 0.05 may be unusually favorable. | Hold the primary comparison fixed, then report a sensitivity sweep over sample spacing and compute cost; use the same prediction times for all methods at each spacing. |
| Incomplete artifact portability | Raw `.npz` and `.pt` files are ignored by git; the published metrics cannot be recomputed without training. | Pin dependency versions, provide a complete command sequence and checksums or an archived dataset, and exclude or clearly label invalid exploratory output. |

The first three rows are the minimum for a convincing empirical paper. A second system is important for a general claim. If resources are limited, present the existing work as a careful Lorenz-63 **case study** and avoid a broad method claim.

## Confirmatory experiment to freeze before running new data

1. Generate a new set of at least 20–24 Lorenz attractor windows separated along a long reference trajectory. Keep the original eight windows as development data. Keep the 4×64 tanh PINN, one-unit window, Adam curriculum, 20,000 steps, and five fixed seeds per window. Record every failure.
2. At prediction times on the 0.05 residual sampling grid with a complete 0.25 trailing interval, compare two features added to the **same** baseline (`log ||r(t)||` and normalized time): (a) log plain mean of sampled residual magnitudes, including the current knot; (b) log mean magnitude after piecewise linear interpolation of the sampled residual **vector**. Target `log E(t+Δ)` at `Δ=0.1` and `0.25`.
3. Primary quantity: paired leave-one-initial-condition-window-out `ΔR² = R²(linear) - R²(box)` on the new windows. Also report log-error MAE, per-window squared-error differences, and uncertainty obtained by resampling **windows**, not time points or individual seeds. Lock these choices before inspecting the new outcomes. Do not tune width, model, or horizon on the confirmatory set.
4. Secondary mechanism analysis: regenerate the Phase 3 tangent-propagator arrays, then predict the error created inside the horizon, `N_Δ = ||e(t+Δ)-Φ(t+Δ,t)e(t)||`, using the same residual features. State that `Φ` and `N_Δ` require the reference solution and are therefore evaluation oracles. Include a current-error oracle baseline to diagnose inherited error.
5. Decision language: call the interpolation-specific finding **replicated** only if the paired improvement is positive at both primary horizons with window-level intervals above zero and the per-window improvements are not concentrated in one or two initial conditions. Otherwise report that temporal aggregation helped but interpolation itself lacked stable incremental evidence. This decision rule is prospective for the new data, not retroactively applied to the development set.

The planned window count is a practical target, **not** a formal power calculation. If the new data remain heterogeneous, report the distribution of window-level effects rather than only a pooled statistic.

## Suggested manuscript structure

1. **Abstract:** question, prespecified negative directional result, exploratory temporal-history result, confirmatory outcome when available, scope.
2. **Introduction:** why residual magnitude at one time need not predict later error in chaotic dynamics; the difference between training weights and post-training diagnostics.
3. **Theory and definitions:** exact nonlinear error equation; linearized Duhamel relation and its approximation limits; definitions of `b₁`, finite-time singular vector `s₁`, pointwise and trailing residual metrics. Distinguish Turinici's scalar time-weight minimax result from this empirical diagnostic question.
4. **Methods:** reference solver accuracy; PINN architecture and training; initial-condition windows; sparse sampling; interpolation and box control; targets; held-out split; cluster unit; declared versus exploratory analyses; artifact provenance.
5. **Results:** linear-saddle sanity check; negative directional result; interpolation fidelity; held-out prediction table and per-window effects; created-error mechanism check; independent validation and second-system results when completed.
6. **Discussion:** why temporal information may matter; why interpolation could help or fail; inherited error; oracle versus deployable quantities; short-window and one-system limits.
7. **Reproducibility appendix:** commands, versions, run counts, acceptance checks, all deviations, and status of the invalid look-ahead output.

Suggested central figures: (i) a time-series example showing residual magnitude and later error, (ii) a diagram of sampled knots, interpolation, and trailing interval, (iii) held-out `R²` gains with window-level intervals and per-window points, and (iv) error-created versus total-error comparison. Keep the original direction-control figure as evidence for the negative result.

## Literature positioning to verify in the manuscript

- [Turinici, *Regime-Aware Time Weighting for Physics-Informed Neural Networks*](https://arxiv.org/abs/2407.21642v2): time weighting **during training** using a scalar growth-rate estimator and a minimax argument. This project studies post-training error indicators.
- [Wang, Sankaran and Perdikaris, *Respecting causality is all you need for training physics-informed neural networks*](https://arxiv.org/abs/2203.07404): causal training for difficult dynamical systems, including Lorenz. Compare the role of time, without presenting our diagnostic as a training method.
- [Hillebrecht and Unger, *Certified machine learning: A posteriori error estimation for physics-informed neural networks*](https://arxiv.org/abs/2203.17055): ODE residual-based error upper bounds. Our empirical predictors are **not certified bounds**.
- [Huseynov, Ahmadova and Bashirov, *Reliable Error Estimation for PINNs: Lower and Upper A Posteriori Bounds*](https://arxiv.org/abs/2606.12050): a 2026 preprint on computable two-sided ODE error enclosures under stated growth and monotonicity assumptions. It reinforces the need to distinguish empirical prediction from certified estimation.

These references establish context, not novelty. Complete a targeted related-work search before submission; new papers or closely related temporal-residual diagnostics may change the novelty framing.

## Immediate writing decision

A paper-style **technical report** can be drafted now with the negative result and the exploratory interpolation finding, each labelled correctly. A stronger manuscript claim about interpolation should wait for the prospective new-window test and created-error mechanism analysis. The current report should not say that interpolation improves PINN training, proves an error bound, or generally works for chaotic systems.
