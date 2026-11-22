# Speaker script: 10 minutes, 4 speakers, 12 slides

Each speaker has about 2.5 minutes, so about 50 seconds per slide. The
words below are a guide; say them in your own words.

---

## Aditya Bhagora: slides 1–3 (the problem and our idea)

**Slide 1, title (about 20 s).**
"Good morning. Our project is about training physics-informed neural networks,
or PINNs, on chaotic systems. We extend a recent paper by Turinici by using the
directions in which errors grow or shrink."

**Slide 2, the problem (about 60 s).**
"A PINN is a neural network that takes time t and outputs a solution x-hat of
t. We train it so that it satisfies the equation dx/dt = f(x). At each time we
measure how much it breaks the equation: the residual r(t) = dx-hat/dt minus
f(x-hat). Training makes the average of r squared small.

The trouble is chaos. In a chaotic system a tiny early mistake grows
exponentially. So a network can have a small residual everywhere and still
follow a completely wrong solution."

**Slide 3, direction matters (about 70 s).**
"Lyapunov exponents tell us how fast small errors grow or shrink. For
Lorenz-63 they are about plus 0.9, 0, and minus 14.6. So in one direction an
error grows, and in another it dies out within about 0.07 time units. The left
plot shows a saddle with one growing and one shrinking direction. The right
plot shows the same small push in each direction: one grows, the other almost
disappears.

We use the finite-time version of these directions: growth and shrinking over
the time window we are solving. Our objective is to weight the training loss
by direction, not only by time. Chandragupt will now explain the reference
paper."

---

## Chandragupt Sharma: slides 4–6 (reference paper and our method)

**Slide 4, Turinici's weighting (about 60 s).**
"Turinici asks: should every time count equally in the loss? He gives each time
a weight rho of t. His first result bounds the final error by the integral of
each residual times its growth factor, exp of the integral of lambda from t to
T. His second result says the best weight is rho of t proportional to exp of
minus the integral of lambda from 0 to t.

So for chaotic systems early times get more weight, and for stable systems
late times do. That is why he calls it regime-aware."

**Slide 5, what we learned (about 50 s).**
"The key idea we keep is that the loss should measure the error a residual
causes, not just its size. But we found three limits:
- it uses one number per time, so all directions are treated the same;
- its estimate of lambda can point the wrong way, and on Lorenz-63 its weights
  grew from 0.26 to 2.75 over the window, the opposite of what it intends;
- it only targets the final time.

In our own code it did not beat equal weights."

**Slide 6, our method (about 60 s).**
"We replace the single number by the full matrix Phi. Phi tells us how a small
push at one time becomes an error later. From it we build the harm matrix G of
t, the integral from t to T of Phi-transpose times Phi. For a residual r at
time t, r-transpose G r measures how much later error it causes, in each
direction.

Our new loss uses r-transpose W r, with W proportional to G divided by the
square root of the trace of G. This comes from the same proof as Turinici's,
and his weight is the special case without direction. The theory guarantees a
gain of at most square root of n over the best no-direction weight. Himanshu
will show an example and how we tested it."

---

## Himanshu Yadav: slides 7–9 (example, testing, main result)

**Slide 7, worked example (about 50 s).**
"Take a saddle with one direction growing at rate plus 1 and one shrinking at
rate minus 16, over the window 0 to 1. The harm along the growing direction is
3.19; along the shrinking direction it is only 0.031. So a residual in the
growing direction is about 100 times more harmful. Equal weights and
Turinici's weights cannot see this; ours can.

We compute G from the network's own current solution every 10 training steps,
so we never need the true solution."

**Slide 8, how we tested (about 45 s).**
"We used six systems, from simple 2D test systems to Lorenz-63, Rössler and
Lorenz-96. For every method and setting we trained 36 networks: 12 starting
points times 3 seeds. Every method used the same network and the same 20,000
training steps; only the weight changed. We compared uniform weights,
Turinici's weights, our harm matrix without direction, and our matrix weight.
We wrote down the hypotheses and decision rules before training, and we call a
network accurate if its error stays below 1."

**Slide 9, main result (about 55 s).**
"Here is the main result. On Rössler over the window 0 to 10, 30 of 36
networks were accurate with our weight, against 13, 4 and 1 for the others, and
the median final-time error was 75 times smaller than with equal weights. On
Lorenz-96 over 0 to 1, our error was 14 times lower than uniform and 62 times
lower than Turinici. Over 0 to 2, we had 10 accurate networks and uniform and
Turinici had none. Our pre-registered test passed for both systems. Pranjul
will show what happens during training."

---

## Pranjul Gupta: slides 10–12 (training curves, the trend, conclusion)

**Slide 10, training curves (about 50 s).**
"This is drawn like Turinici's figures. On the left is the error during
training for Rössler: with our weight it keeps falling, about 100 times, while
the others stall around 2 to 4. On the right is one example run against the
true solution. Our network follows it over the whole window; the others leave
it near t = 3.5 and never come back."

**Slide 11, more harmless directions (about 50 s).**
"To test the theory, we added extra fast-shrinking variables to Lorenz-63. Our
matrix weight stays at 27 of 36 accurate for every number of extra variables.
The weight without direction drops from 24 to 5, and uniform drops to 0. The
advantage of direction grows from 2 times to 100 times, which is exactly what
the theory predicts."

**Slide 12, limits and conclusion (about 50 s).**
"To be honest about limits: on plain Lorenz-63 most of the gain comes from
timing, and a well-tuned causal method does about as well. On easy problems our
weight costs a little accuracy, and it needs about 3 times more computing time.

In conclusion, using one matrix per time instead of one number gives a
direction-aware weight that needs no tuning. It clearly improved training on
Rössler and Lorenz-96. Thank you; we are happy to take questions."

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
- **"Is the result luck?"**
  Each setting used 36 networks, the rules were fixed before training, and a
  complete repeat on different hardware gave the same counts.
