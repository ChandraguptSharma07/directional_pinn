# Speaker script: 10 minutes, 4 speakers, 12 slides

The deck is `presentation.pptx` (also `presentation.pdf`). The same text is in
the speaker notes of each slide. Each speaker has about 2.5 minutes, and the
times below add up to about 10 minutes. Treat the words as a guide and say them
in your own words.

| Speaker | Slides | Topic |
|---|---|---|
| Aditya Bhagora | 1–3 | the problem and the key idea |
| Chandragupt Sharma | 4–6 | the reference paper and our method |
| Himanshu Yadav | 7–9 | worked example, how we tested, main result |
| Pranjul Gupta | 10–12 | training curves, the trend, limits and conclusion |

---

## Aditya Bhagora: slides 1–3

**Slide 1, title (about 20 s).**
"Good morning. Our project is about training physics-informed neural networks,
or PINNs, on chaotic systems. We extend a recent paper by Turinici: instead of
giving each time one weight, we give each time a matrix built from the
directions in which errors grow or shrink."

**Slide 2, the problem (about 60 s).**
"A PINN is a neural network that takes time t and outputs a solution x-hat of
t. We train it so that it satisfies the equation dx/dt = f(x); the starting
value x0 is built into the network.

At each time we measure how much it breaks the equation: the residual
r(t) = dx-hat/dt minus f(x-hat). Training makes the average of the squared
residual small over 512 time points.

The trouble is chaos. In a chaotic system a tiny early mistake grows
exponentially. So a network can have a small residual everywhere and still
follow a completely wrong solution."

**Slide 3, direction matters (about 70 s).**
"Lyapunov exponents tell us how fast small errors grow or shrink. For
Lorenz-63 they are about plus 0.9, 0 and minus 14.6.

The plot shows a simpler saddle with rates plus 1 and minus 16. On the left,
one direction pushes away and the other pulls in. On the right, the same small
push is followed for one time unit: along the growing direction it grows about
2.7 times; along the shrinking direction it shrinks about 9 million times.

We use the finite-time version of these directions, over the window we are
solving. Our goal is to weight the training loss by direction, not only by
time. Chandragupt will now explain the reference paper."

---

## Chandragupt Sharma: slides 4–6

**Slide 4, Turinici's time weighting (about 60 s).**
"Our reference paper is by Turinici. He asks: should every time count equally
in the loss? He gives each time a weight rho of t.

His Proposition 1 bounds the final error: each residual is multiplied by its
growth factor, exp of the integral of lambda from t to T, before it reaches the
end. His Proposition 2 says the best weight is rho of t proportional to exp of
minus the integral of lambda from 0 to t.

The left plot shows this weight: for a chaotic system early times get more
weight, for a stable system late times do. The right plot shows why: a mistake
made early grows the most by the end. That is why he calls it regime-aware, and
the weight needs no tuning."

**Slide 5, what we learned (about 50 s).**
"The idea we keep is that the loss should measure the error a residual causes,
not just its size.

But we found limits. First, it uses one number per time, so every direction is
treated the same. Second, its estimate of lambda, the inner product of f(U)
minus f(0) with U, divided by the length of U squared, can point the wrong way:
on Lorenz-63 its weight rose from 0.26 to 2.75 over the window, the opposite of
what it should do. Third, it only targets the error at the final time.

In our tests it did not beat equal weights."

**Slide 6, our method (about 60 s).**
"We replace the single number by the full matrix Phi, the tangent propagator:
it tells us how a small push at time s becomes an error at a later time t.
Adding up all the pushes gives the error: e(t) is approximately minus the
integral of Phi times r.

From Phi we build the harm matrix G(t), the integral from t to T of
Phi-transpose times Phi. For a residual r at time t, r-transpose G r measures
the total squared error it causes later. Its eigenvectors are the finite-time
Lyapunov directions.

Our new loss is the average of r-transpose W r, with W proportional to G
divided by the square root of its trace. We get it by repeating Turinici's
reasoning with matrices; his weight is the special case without direction. In
the worst case, the error bound improves by a factor between 1 and the square
root of n. Himanshu will show an example and how we tested it."

---

## Himanshu Yadav: slides 7–9

**Slide 7, worked example (about 50 s).**
"Take a saddle with one direction growing at rate plus 1 and one shrinking at
rate minus 16, on the window from 0 to 1. The harm along the growing direction
is the integral of exp(2 tau), which is about 3.19. Along the shrinking
direction it is only about 0.031. So a residual along the growing direction is
about 100 times more harmful. Equal weights and Turinici's weights cannot see
this; ours can.

During training we compute everything from the network's own current
solution: the tangent propagator along it, then G by a short backward sweep
over the time grid, then the weight. We refresh it every 10 training steps, so
we never need the true solution."

**Slide 8, how we tested (about 45 s).**
"We used six systems, from simple two-dimensional test systems to Lorenz-63,
Rössler and Lorenz-96. We compared four methods: uniform weights, Turinici's
weights, our harm matrix without direction, which we call scalar, and our
matrix weight.

For every method and setting we trained 36 networks, 12 starting points times
3 seeds, or 24 for the two-dimensional systems. Every method used the same
network and the same 20,000 training steps; only the weight changed.

We wrote down the hypotheses and decision rules before training, and we call a
network accurate if its error stays below 1 over the whole window."

**Slide 9, main result (about 55 s).**
"Here is the main result: how many of 36 networks learned the correct
solution.

On Rössler over the window 0 to 10, 30 networks were accurate with our weight,
against 13, 4 and 1 for the others, and the median final-time error was 75
times smaller than with equal weights.

On Lorenz-96 over 0 to 1, our error was 14 times lower than uniform and 62
times lower than Turinici. Over 0 to 2, we had 10 accurate networks while
uniform and Turinici had none.

Our pre-registered hypothesis H4 was supported for both systems. Pranjul will
show what happens during training."

---

## Pranjul Gupta: slides 10–12

**Slide 10, training curves (about 50 s).**
"This figure is drawn like Turinici's figures, for Rössler over the window 0
to 10.

On the left is the median error of the 36 networks during training: with our
weight it keeps falling, from about 10 to about 0.03, while the others stall
between 2 and 4.

On the right is one example run against the true solution. Our network follows
it over the whole window; the others leave it near t = 3.5 and never come
back."

**Slide 11, more harmless directions (about 50 s).**
"To test the theory, we added k extra fast-shrinking variables to Lorenz-63:
0, 3, 9 and 29. These add harmless directions.

Our matrix weight stays at 27 of 36 accurate networks for every k. The scalar
weight without direction drops from 24 to 5, and uniform drops from 7 to 0.

Compared with the scalar weight, the error of our weight is about 2 times lower
with no extra variables, which is not statistically significant, then 8, 50
and 100 times lower. More harmless directions give a bigger benefit from using
direction, which is what the theory predicts."

**Slide 12, limits and conclusion (about 50 s).**
"To be honest about limits: on plain Lorenz-63 most of the gain comes from
timing, and a well-tuned causal training method does about as well. On easy
problems our weight costs a little accuracy, and it needs about 3 times more
computing time.

In conclusion, using one matrix per time instead of one number gives a
direction-aware weight that needs no tuning. On Rössler and Lorenz-96 it beat
uniform weights, Turinici's weights and the weight without direction.

Thank you; we are happy to take questions."

---

## Likely questions, with short answers

- **"Why not just use a normal ODE solver?"**
  For plain forward problems a classical solver is faster. Our work improves
  how PINNs are trained. That matters where PINNs are used, for example when
  data must be fitted too.
- **"Are these real Lyapunov vectors?"**
  They are finite-time directions over the current window. They approach the
  long-time Lyapunov directions only when the window is long.
- **"How do you get the directions without the true solution?"**
  From the network's own current solution, updated every 10 training steps.
- **"Why divide by the square root of the trace?"**
  It is the weight that gives the smallest worst-case error bound, by the same
  argument as Turinici's Proposition 2, done with matrices.
- **"Is the result luck?"**
  Each setting used 36 networks and the rules were fixed before training. We
  also repeated the Lorenz-63 runs on different hardware and got the same
  success counts.
