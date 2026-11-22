# Direction-aware (matrix) residual weighting for PINNs

This folder holds a follow-up to the directional study in the rest of the
repository. That study asked whether the direction of a PINN residual
*predicts* later error. This one asks whether direction can *improve
training*.

[Turinici (arXiv:2407.21642v2)](https://arxiv.org/abs/2407.21642v2) weights
the residual in time with one number per time. Here the weight is a matrix
per time, built from the tangent propagator of the dynamics.

- [`docs/plan.md`](docs/plan.md): the hypotheses, methods, gates and decision
  rules. They were committed before any training.
- [`docs/theory.md`](docs/theory.md): the derivation. The best matrix weight
  is G(t)/sqrt(tr G(t)), where G is the window Gramian of the tangent
  propagator. Turinici's weight is the direction-blind special case. The
  guaranteed gain is at most sqrt(n).
- [`docs/progress.md`](docs/progress.md): the dated record of every step,
  check and deviation.
- `results/<gate>/analysis.md`: the planned analyses, once runs finish.

## Layout

| path | content |
|---|---|
| `mw/systems.py` | linear saddle, Davis–Skodje, Lorenz-63 (optionally with damped modes): right-hand sides and Jacobians in NumPy and PyTorch |
| `mw/model.py` | a batch of independent PINNs with exact initial state and forward-mode time derivative |
| `mw/weights.py` | step propagators, window Gramian, final-time Cauchy–Green tensor, matrix / scalar / Turinici / causal weights |
| `mw/train.py` | training loop and metrics |
| `mw/reference.py` | references (DOP853, rtol = atol = 1e-12) and Lorenz attractor states |
| `experiments/make_jobs.py` | writes the frozen job files in `jobs/` |
| `experiments/run_job.py`, `experiments/pool.py` | run one job, or a directory of jobs with a worker pool |
| `experiments/analyze.py` | the planned contrasts and decisions |
| `remote/kaggle_kernel.py` | packs jobs into a private Kaggle GPU kernel |
| `tests/test_mw.py` | tests (`python -m pytest tests`) |

## Running

```
python experiments/make_jobs.py gate1                 # job files (already committed)
python experiments/pool.py jobs/gate1 --workers 10    # CPU
python experiments/pool.py jobs/gate2 --device cuda   # GPU
python experiments/analyze.py gate1
```

Each job writes `results/.../summary.json`, which is committed. It also
writes `data.npz`, which git ignores and the jobs regenerate.
