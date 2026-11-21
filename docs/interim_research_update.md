# Interim research update: residual direction versus residual history

**21 November 2026 · Lorenz-63 PINNs · presentation brief**

## Research question

Can a trained PINN's differential-equation residual indicate its future trajectory error? The original project tested whether the residual's **direction** relative to expanding dynamics adds information beyond its instantaneous magnitude. That preset test did not support a benefit on held-out Lorenz-63 initial conditions. The submitted report proposed interpolation of the residual between sampled times as a follow-up.

## What we tested next

We rebuilt 40 standard PINNs (eight initial conditions, five seeds), sampled each residual vector every 0.05 time units, and predicted future total trajectory error at sampled times. The same leave-one-initial-condition-out split compared:

| Method | Feature added to a pointwise-residual-plus-time baseline |
|---|---|
| Plain sampled average | Mean residual magnitude at knots in the preceding 0.25 time units, including the current knot |
| Interpolated average | Mean magnitude of the piecewise linearly interpolated residual vector over the same interval |

| Future horizon | Baseline held-out R² | Plain average held-out R² | Interpolated held-out R² | Paired interpolation gain (95% window bootstrap) |
|---:|---:|---:|---:|---:|
| 0.10 | 0.279 | 0.691 | 0.750 | +0.060 (+0.022, +0.110) |
| 0.25 | 0.568 | 0.828 | 0.880 | +0.052 (+0.032, +0.117) |

**Finding:** Recent residual magnitude is more informative than a residual at one instant for these short-horizon predictions. Interpolation adds a smaller gain over a plain average of the same sparse samples. At horizon 0.5, the interpolation feature did not reliably improve over the pointwise baseline. These are exploratory results from eight independent initial-condition windows; the plain-average control was added after examining earlier results.

## How the submitted report's methods fit

Interpolation and simple averaging are now compared on the **same diagnostic task**. The Duhamel-weighted integral and Gaussian-process smoothing remain proposed diagnostic methods. Turinici-style time weighting changes the **PINN training loss** and needs a separate training comparison; it cannot be placed in the same post-training feature table.

## Next decisive experiment

Freeze the interpolation-versus-plain-average comparison and test it on new, untouched Lorenz windows. Also evaluate the error **created within** each prediction horizon, which separates new residual-driven error from error inherited from the past. Then test a second dynamical system before making a general claim. See [the paper roadmap](paper_roadmap.md) and [full run record](interpolation_progress.md).

The PDF version of this brief is [interim_research_update.pdf](interim_research_update.pdf). The source data are in `results/interpolation_rebuild/interpolation_box_control/metrics.json`.
