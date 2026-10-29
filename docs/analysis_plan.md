# Analysis plan for the central experiment (fixed before the main runs)

This plan was written and committed before any relationship between PINN
residuals and PINN errors was looked at. Its purpose is to make the
conclusion depend on rules chosen in advance, not on the results. Deviations
from it, if any, are reported in the results document together with the
reason.

## Data

* **Initial conditions:** analysis windows 1–8 of Phase 2 (window 0 was used
  only for the pilot).
* **PINNs:** for each window, 5 independent trainings (seeds 0–4) with the
  settings chosen by the pilot (Phase 4): 40 runs. All runs are reported,
  including failed ones.
* **Grid:** reference grid $t_j = jh$, $h = 0.002$, on the PINN window $[0, T]$.
  Statistics use every 5th point ($\Delta = 0.01$) to limit redundancy; the
  uncertainty estimates treat runs, not time points, as the independent units.
* **Horizons:** $\Delta t \in \{0.05, 0.1, 0.25, 0.5, 1\}$. For horizon $\Delta t$, the
  usable times are $t \in [0, T - \Delta t]$ (the PINN is not defined beyond $T$).

## Quantities

For each run, time $t$ and horizon $\Delta t$:

* error vector $e(t) = \hat x(t) - x(t)$, error size $E(t) = \lVert e(t)\rVert$;
* residual vector $r(t)$ (from automatic differentiation, evaluated in float64);
* **targets:**
  * primary: the future error $E_\Delta(t) = \lVert e(t+\Delta t)\rVert$, as in the
    original question;
  * secondary (mechanistic): the error created inside the horizon,
    $N_\Delta(t) = \lVert e(t+\Delta t) - \Phi(t+\Delta t, t)\,e(t)\rVert$, i.e. the
    future error minus the linear propagation of the error already present at $t$;
* **directions** (oracle: computed from the reference trajectory):
  * $b_1(t)$, the backward (QR) Lyapunov vector;
  * $s_1(t;\Delta t)$, the leading right singular vector of $\Phi(t+\Delta t, t)$;
* **predictors:** $R = \lVert r\rVert$, $R_u = |r\cdot u|$,
  $R_\perp = \lVert r - (r\cdot u)u\rVert$, alignment $A_u = R_u / (R + 10^{-12})$, for
  $u \in \{b_1, s_1\}$; secondary, the oracle amplification
  $R_\Phi = \lVert\Phi(t+\Delta t, t)\, r(t)\rVert$, which uses all directions at once;
* **controls (Phase 8):** the same predictors with $u$ replaced by
  (a) independent uniformly random unit vectors at every time (20 draws),
  (b) $b_1$ or $s_1$ shuffled in time within each run (20 permutations),
  (c) the least-expanding directions $b_3$ and $s_3(t;\Delta t)$,
  (d) the flow direction $f(x)/\lVert f(x)\rVert$ (the neutral direction).

## Analyses

1. **Correlations** per run: Pearson correlation of logarithms and Spearman
   correlation between each of $R$, $R_u$, $R_\perp$ and $E_\Delta$ (and $N_\Delta$).
   Reported per run, as mean and standard deviation over runs, and pooled.
   Log–log scatter plots and binned medians.
2. **Orientation at fixed magnitude:** within each run, the times are split
   into quintiles of $R$, and each quintile into terciles of $A_u$. The effect is
   the mean over quintiles of $\log$ median $E_\Delta$ (high tercile) minus $\log$
   median $E_\Delta$ (low tercile). Also the partial Spearman correlation
   between $A_u$ and $E_\Delta$ given $R$.
3. **Held-out prediction** (the decisive test). Ordinary least squares on
   $y = \log E_\Delta$:
   * M0: $y = a + b\log R$
   * M1(u): $y = a + b\log R + c\,A_u$
   * M2(u): $y = a + b\log R + c\log(R_u + 10^{-6}\,\overline{R})$, with
     $\overline{R}$ the median residual size of the run
   * M3: $y = a + b\log R + c\log(R_\Phi / R)$ (oracle amplification)

   Evaluation is leave-one-window-out: fit on the runs of 7 windows, predict
   the runs of the held-out window, repeat for all 8 windows. Reported:
   out-of-sample $R^2$, mean absolute error and Spearman correlation of the
   pooled out-of-fold predictions. The key number is
   $\Delta R^2 = R^2(\text{M1}) - R^2(\text{M0})$. Its 95 % interval comes from a
   cluster bootstrap over runs (2000 resamples of the out-of-fold predictions).

## Decision rule

Directional information is called **supported** for a direction
$u \in \{b_1, s_1\}$ if, for at least 3 of the 5 horizons, all of the following
hold for the primary target $E_\Delta$:

1. $\Delta R^2 \ge 0.02$ and its 95 % bootstrap interval excludes 0;
2. the fitted coefficient $c$ of $A_u$ in M1 (all data) is positive, i.e. more
   alignment with the expanding direction goes with larger future error;
3. $\Delta R^2$ for $u$ exceeds the 95th percentile of $\Delta R^2$ over the 20
   random-direction draws and over the 20 time-shuffled versions of $u$;
4. the fixed-magnitude alignment effect (analysis 2) is positive, with a 95 %
   cluster-bootstrap interval excluding 0.

If neither $b_1$ nor $s_1$ meets the rule, the hypothesis is **not supported**.
If some but not all conditions hold, the result is reported as
**inconclusive**. Phase 9 (a directional training loss) is carried out only if
the result is "supported". Two directions are tested, so a "supported"
result for only one of them is reported with that multiplicity in mind.

## Sensitivity analyses (reported, not part of the decision)

* secondary target $N_\Delta$;
* restriction to the near-linear regime: samples with $E(t) < 1$ and $E(t+\Delta t) < 1$;
* checking the linear error picture: the Duhamel reconstruction
  $e(t) \approx \Phi(t, 0)\,e(0) + \int_0^t \Phi(t, s)\,r(s)\,ds$ against the actual error;
* precision: the primary analysis repeated on float64-trained PINNs (one seed
  per window);
* per-window and per-seed results, so that no single run drives the conclusion.
