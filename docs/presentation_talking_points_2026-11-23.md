# Speaking notes for the 23 November presentation

These notes match [`presentation_2026-11-23.pdf`](presentation_2026-11-23.pdf).
They distinguish measured results from explanations that remain hypotheses.

## Slide 1 — The question (about 40 seconds)

“The project asked whether a physics-informed neural network's equation
residual carries more information than its size alone. The hypothesis was
that a residual pointing along an expanding Lorenz direction would lead to
larger later trajectory error. The study tested alignment with a backward
Lyapunov vector and a leading finite-time singular vector. Its planned
40-run, held-out analysis found that neither alignment met the decision
rule. Turinici's reference paper addresses a related but different question:
how to weight equation violations over time during training.”

## Slide 2 — The model bottleneck and improvement (about 45 seconds)

“The original single PINN was reliable only for roughly one time unit, so
the project built a sequential model: eight short PINNs each cover 0.25
time units, and each receives the preceding model's *predicted* endpoint.
No true state from inside the window is used to train it. On the development
set, 35 of 40 runs stayed below error 1 throughout two time units. More
importantly, the frozen model was then tested on fresh initial conditions:
30 of 40 stayed below error 1 throughout, and 32 of 40 for at least 90%
of the window. The failures are shown in the plot, especially window 1.
Thus the method extends accuracy, but it is not uniformly reliable.”

## Slide 3 — The hypothesis on new windows (about 45 seconds)

“The original directional test was repeated on all 40 fresh runs, holding
out one initial-condition window at a time and excluding exact segment
joins. Neither one-vector alignment helped, even when attention was
restricted to portions where the model stayed near the true trajectory.
A full-propagator quantity did add held-out information at longer horizons;
the plotted intervals resample whole initial-condition windows. This is
only an *oracle*: it uses the known reference trajectory to compute the
propagator and cannot yet be used to diagnose an unknown solution.”

## Slide 4 — Conclusion (about 30 seconds)

“The result is a bounded negative finding: simple pointwise alignment with
one expanding direction is not a reliable predictor of future error for
these Lorenz PINNs. The sequential model gives a better and longer test
bed, while the oracle result suggests that full dynamical amplification
and residual history matter. The next research step is to test a causal
predictor built only from past PINN residuals against simple baselines,
then study another dynamical system.”

## If the professor asks

**Did the hypothesis fail?** The tested pointwise one-vector version was not
supported under the preset rule, and the frozen fresh-window repeat gave the
same result. This does not prove that residual direction never matters.

**Why?** The trained residuals often already align with the leading
direction, so that feature has little variation. Total future error also
contains error accumulated earlier. These are plausible explanations;
the experiments do not isolate a unique cause.

**Is the new model a solution to chaos?** No. It is accurate throughout
two units in 30/40 fresh runs, with clustered failures on difficult windows.

**What is ready now?** The project has a presentation-ready Lorenz-specific
conclusion, numerical validation, and an explicitly limited oracle
mechanism. A practical reference-free predictor remains open.
