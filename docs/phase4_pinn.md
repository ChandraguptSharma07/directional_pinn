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

Pilot results and the chosen values are recorded in
`results/pinn_pilot/metrics.json` and summarised in the results document.
