# Phase 4: the baseline PINN

Code: `src/pinn/model.py`, `src/pinn/residual.py`, `src/pinn/train.py`,
`src/pinn/runs.py`, `experiments/05_pinn_pilot.py`, `experiments/06_pinn_runs.py`.

## Model

A fully connected network maps time to the state. Time enters as
$\tau = 2t/T - 1 \in [-1, 1]$ on the window $[0, T]$, and the network's three
outputs are rescaled to the attractor's size,
$\hat x(t) = (0, 0, 25) + 10\cdot\mathrm{net}(\tau)$. Defaults: 4 hidden layers of 64
units, $\tanh$ activations, Xavier-normal weights and zero biases. All of these
are configurable.

## Residual

$$
r(t) = \frac{d\hat x}{dt} - f(\hat x(t)),\qquad
f(x, y, z) = \big(\sigma(y-x),\ x(\rho-z)-y,\ xy-\beta z\big).
$$

The time derivative comes from automatic differentiation: each output component
is differentiated with respect to the input time. The tests check the residual
on exact solutions of the linear saddle (residual 0), on a perturbed solution
(known nonzero residual), on a Lorenz fixed point (residual 0), on a polynomial
path (residual computed by hand), and the network's time derivative against
finite differences. After training, residuals are evaluated with the trained
weights in float64, so rounding in the evaluation does not add noise.

## Loss and training

$$
L = \frac1N\sum_{i=1}^N \lVert r(t_i)\rVert^2 + \lambda_{IC}\,\lVert\hat x(0) - x_0\rVert^2 .
$$

* The $N = 1024$ collocation times are redrawn at every iteration by stratified
  sampling: one uniform random time in each of $N$ equal bins of $[0, T]$.
* Adam, 20 000 iterations, learning rate decaying exponentially from
  $10^{-3}$ to $10^{-5}$.
* float32, one CPU thread per run (runs execute in parallel processes). A run
  is deterministic for a given seed: the unit tests train twice with the same
  seed and get identical losses and weights.
* **Time curriculum** (added after the first pilot, see below): during the first
  60 % of the iterations the collocation times are drawn from $[0, T_i]$, with
  $T_i$ growing linearly from 0.25 to $T$; afterwards from $[0, T]$. The loss is
  the standard loss at every iteration (only on a growing interval), and at
  the end of training it is exactly $L$ on $[0, T]$.
* The seeds of a window are trained together as one stacked model (batched
  matrix products). The models share nothing, and Adam updates every
  parameter from its own gradient history, so this is the same as training
  them one by one; a unit test confirms agreement to $10^{-9}$ in float64.
* No directional term: this is the standard PINN loss. L-BFGS was not added.

## Choosing the window length and the initial-condition weight (pilot)

Window 0 is reserved for this choice and never enters the analysis. The rule
was fixed before the pilot ran and looks only at PINN accuracy, never at
residual directions:

1. $\lambda_{IC} \in \{1, 10, 100\}$: the value with the smallest median error
   $\lVert\hat x(t) - x(t)\rVert$, averaged over the candidate window lengths;
2. $T \in \{2, 3, 4\}$: the longest window for which that PINN keeps the error below
   1 (about 5 % of the attractor's size) for at least 90 % of the window, so that
   most of the data stay where the linear error picture of Phase 0 applies.

### First pilot: plain training fails

Without a curriculum, every PINN on windows of length 2 and 3 (all
$\lambda_{IC} \in \{1, 10, 100\}$) converged to the same wrong answer:
$\hat x \approx \hat y \approx 0$ with $\hat z$ decaying, which is the exact Lorenz
solution on the invariant $z$ axis. Its residual is nearly zero except for a
short jump right after $t = 0$, so it is a cheap minimum of the loss; the error
is the size of the attractor (median 17.5 for $T = 2$, 22.0 for $T = 3$). The
windows of length 4 were stopped before finishing. Such a PINN cannot test the
hypothesis, because its error does not come from small residuals being
amplified. These runs are kept in `results/pinn_pilot_attempt1/` (weights and
arrays are not tracked in git; the summaries are in the run directories).

### Second pilot: with the time curriculum

Candidates $T \in \{1, 2, 3, 4\}$ and $\lambda_{IC} \in \{10, 100\}$ (the first pilot showed
initial-condition errors of about 2 for $\lambda_{IC} = 1$, so that value was
dropped); same selection rule. (This pilot was restarted once after the
machine rebooted; the restart reran it from scratch.)

| $T$ | $\lambda_{IC}$ | median error | max error | share of window with error < 1 | median $\lVert r\rVert$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 10  | 0.83  | 1.7  | 65 % | 0.57 |
| 1 | 100 | 0.086 | 0.20 | 100 % | 0.11 |
| 2 | 10  | 6.1   | 35.7 | 10 % | 0.69 |
| 2 | 100 | 17.5  | 40.6 | 3 % | 0.083 |
| 3 | 10  | 12.6  | 32.4 | 4 % | 3.7 |
| 3 | 100 | 22.0  | 45.9 | 2 % | 0.12 |
| 4 | 10  | 15.1  | 33.9 | 3 % | 1.5 |
| 4 | 100 | 22.5  | 45.9 | 2 % | 0.12 |

Even with the curriculum, the standard PINN follows the trajectory only over
$T = 1$, roughly one Lyapunov time. For $T \geq 2$ it either collapses again onto
the $z$-axis solution ($\lambda_{IC} = 100$: tiny residual, huge error) or fails
less cleanly ($\lambda_{IC} = 10$).

**Deviation from the pre-set rule.** Applied literally, the rule first picks
$\lambda_{IC} = 10$, because its error averaged over all four window lengths is
smaller. That average is dominated by the long windows, where every PINN fails.
The rule then finds no window in which that PINN stays accurate and falls
back to the shortest, $T = 1$, where $\lambda_{IC} = 10$ is ten times less accurate
than $\lambda_{IC} = 100$. This contradicts the rule's purpose of picking the most
accurate setting. The main runs therefore use $T = 1$ and $\lambda_{IC} = 100$,
the most accurate setting at the only workable window length. The rule's
literal choice ($T = 1$, $\lambda_{IC} = 10$) is run as a complete second set and
analysed the same way, so the conclusion can be checked against it. Both
choices were made from PINN accuracy alone, before any residual direction was
examined.

**Consequence for the analysis.** With $T = 1$, the horizon $\Delta t = 1$ has no usable
time points ($t + \Delta t \leq T$ leaves only $t = 0$), so the decision rule is applied
to the four horizons $0.05, 0.1, 0.25, 0.5$, still requiring 3 that pass.
