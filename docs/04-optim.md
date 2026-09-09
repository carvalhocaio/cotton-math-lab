# 04 — Optimization

Six first-order optimizers, same interface (`step()`/`zero_grad()`
from Module 2's `SGD`), each validated step by step against the
equivalent `torch.optim` — not just "it converges", identical trajectory. Then,
Newton and BFGS, reusing `gradient()` and `hessian()` from Module 2 with
no new math. The throughline: each method trades one pain for
another, and the right pain depends on the problem's geometry, not on which method
is trendy.

---

## Momentum: accumulates direction, but amplifies the effective step

$$
v \leftarrow \text{momentum} \cdot v + \nabla\theta, \qquad \theta \leftarrow \theta - \text{lr} \cdot v
$$

Validated against `torch.optim.SGD(momentum=...)`, identical trajectory.

### The finding that didn't follow the expected script

First attempt, same `lr` for SGD and Momentum on the surface
$f(x,y)=x^2+10y^2$: Momentum ended up **786× worse** than plain SGD. Not a bug —
Momentum amplifies the effective step by a factor close to $1/(1-\text{momentum})$
(with momentum=0.9, that's 10×), and this factor can destabilize the
direction of highest curvature even when the nominal `lr` seemed
reasonable. Varying `lr`:

| lr | SGD | Momentum | ratio |
|---|---|---|---|
| 0.005 | 4.05 | 1.38 | 0.34 |
| 0.02 | 0.344 | **0.012** | **0.034** (29× better) |
| 0.03 | 0.064 | 0.118 | 1.86 (Momentum already worse) |

The advantage exists, and it's large — but within a narrow window of `lr`, not
universally. "Momentum accelerates convergence" is conditionally true, not
unconditionally.

---

## Nesterov: the same idea, looking ahead

$$
v \leftarrow \text{momentum}\cdot v + \nabla\theta, \qquad \theta \leftarrow \theta - \text{lr}\cdot(\nabla\theta + \text{momentum}\cdot v)
$$

PyTorch (and this implementation) uses an algebraic reformulation that avoids
a second forward evaluation at a future position — same result, without the
extra cost. Validated against `torch.optim.SGD(nesterov=True)`, difference
$6.8\times10^{-8}$.

### Actually resolves the previous cycle's instability

At the same `lr=0.03` where classic Momentum ended up worse than SGD (0.118 vs
0.064), Nesterov reached **0.0103** — better than both. At `lr=0.04`,
Momentum gets even worse (0.768); Nesterov stays stable (0.0007). The
"look-ahead" correction shifts the stability boundary — it's not
just theoretical elegance.

---

## RMSProp: changes axis — scale, not direction

$$
v \leftarrow \alpha v + (1-\alpha)(\nabla\theta)^2, \qquad \theta \leftarrow \theta - \text{lr}\cdot\frac{\nabla\theta}{\sqrt{v}+\epsilon}
$$

Validated against `torch.optim.RMSprop`, difference $\sim10^{-8}$.

### The proof that it equalizes scales, with a number

On the surface with 10× more curvature in $y$ than in $x$: with SGD, the step
in $y$ was **6.8× larger** than in $x$ — curvature directly turns into an
unequal step. With RMSProp, the ratio dropped to **1.000**. Both parameters
advance equally, despite the drastically different curvature — per-parameter
adaptation actually working, not just a claim from a manual.

Practical consequence: RMSProp converges well over a **much**
wider `lr` range than Momentum needed (tested 0.1 to 0.3, all converging to
loss $<10^{-4}$) — no need to hunt for the right `lr` the same way.

---

## Adam: both axes together, plus bias correction

$$
m \leftarrow \beta_1 m + (1-\beta_1)\nabla\theta, \quad v \leftarrow \beta_2 v + (1-\beta_2)(\nabla\theta)^2
$$
$$
\hat{m} = \frac{m}{1-\beta_1^t}, \quad \hat{v} = \frac{v}{1-\beta_2^t}, \quad \theta \leftarrow \theta - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}
$$

Validated against `torch.optim.Adam`, difference $\sim10^{-7}$.

### Why bias correction exists: an exact invariant

At $t=1$, the correction exactly cancels the introduced bias factor:
$\hat{m}=\nabla\theta$ and $\hat{v}=(\nabla\theta)^2$, so
$\hat{m}/\sqrt{\hat{v}} = \text{sign}(\nabla\theta)$ — **Adam's first
step is always $\pm\text{lr}$, regardless of the initial gradient's
magnitude**. Tested with gradients from 0.001 to 500: all gave a step
$\approx\text{lr}$, to the fourth decimal place.

Without the correction, the first step ends up inflated by a factor
$(1-\beta_1)/\sqrt{1-\beta_2}$ — with the default betas, $\approx3.16\times$
larger than it should be. The square root only applies to the $v$ side (which carries
$(\nabla\theta)^2$); the $m$ side has no root at all — this asymmetry between
how $\beta_1$ and $\beta_2$ bias their respective averages is exactly what
the correction fixes.

---

## AdamW: why it isn't "Adam + weight decay"

$$
\theta \leftarrow \theta - \text{lr}\cdot\left(\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} + \text{weight\_decay}\cdot\theta\right)
$$

Validated against `torch.optim.AdamW`, difference $\sim10^{-6}$.

### The demonstration with two parameters of different history

Two parameters, same initial value, radically different gradient
histories (one receives a gradient 10,000× larger than the other for 20 steps),
then the real gradient is zeroed — only the `weight_decay` effect is observed:

| Method | relative decay (large history) | relative decay (small history) |
|---|---|---|
| AdamW | 56.93% | 56.93% |
| "Adam + L2 in the gradient" | 43.6% | 68.6% |

With AdamW, both decay by **the same proportion** — the nominal
`weight_decay` is the real regularization strength, full stop. With the naive
form (adding `weight_decay·θ` to the gradient before Adam processes it), the decay
term enters the moving averages and ends up rescaled by the
per-parameter adaptation: the parameter with a large $v$ (large gradient
history) receives **less** effective decay; the one with a small $v$ receives
**more**. The regularization strength ends up depending on each
parameter's training history — not what anyone intends when choosing a
`weight_decay=0.01`.

`AdamW` inherits from `Adam` in code (only `step()` changes) — the only
time in the module where one optimizer reuses another via inheritance, because here the
relationship genuinely is "the same thing, plus one term", unlike
Momentum→Nesterov or RMSProp→Adam, which are distinct rule families.

---

## Rosenbrock: the benchmark that debunks the "Adam is always better" cliché

$$
f(x,y) = (1-x)^2 + 100(y-x^2)^2, \quad \text{minimum at } (1,1)
$$

A narrow, **curved** valley (follows the parabola $y=x^2$) — unlike
the ill-conditioned quadratic from the previous cycles, whose principal axes
coincide with $x$ and $y$. The difficulty here isn't unequal scale,
it's a curved trajectory.

Running all six, same budget of 2000 steps, `lr` tuned per method:

| Optimizer | distance to the minimum |
|---|---|
| SGD | 0.82 (didn't move) |
| **Momentum** | **0.0003** |
| **Nesterov** | **0.0002** |
| RMSProp | 0.12 |
| Adam | 0.19 |

Momentum and Nesterov **beat** the adaptive methods, by a large margin.
The reason: at the bottom of the valley, the gradient consistently points in
the same general direction for many steps in a row — exactly the regime
where accumulating direction (Momentum) wins over normalizing scale per
coordinate (Adam/RMSProp), because here the challenge was never unequal scale
between axes. "Adam is always the safe choice" is folklore, not a theorem — the
problem's geometry decides, and sometimes it decides against what every
deep learning blog suggests off the top of its head.

---

## Newton and BFGS: fast in iterations, expensive per iteration

$$
\theta \leftarrow \theta - H^{-1}\nabla\theta
$$

Directly reuses `gradient()` and `hessian()` from Module 2 — no new
math.

### An exact invariant

On a pure quadratic, the Hessian is constant: the Newton step **is** the
minimum, not an approximation of it. Tested: converges in exactly 1
iteration, result exact to $10^{-8}$.

### On Rosenbrock: 7 iterations against 2000

Newton converged on Rosenbrock in **7 iterations**, down to machine
precision ($3.5\times10^{-16}$ of distance). Momentum needed 2000
steps to reach $0{.}0003$. Second order "knows" the local curvature and
doesn't need to feel its way — but each iteration costs assembling and solving an
$n\times n$ system.

BFGS (quasi-Newton, approximates $H^{-1}$ using only gradients, never
assembling the true Hessian) converged in 35 iterations — more than Newton, fewer
than thousands of first-order steps, and validated against
`scipy.optimize.minimize(method='BFGS')`, difference $<10^{-4}$.

### Why it "dies" at high dimension — measured, not assumed

Count of `gradient()` calls needed to assemble the Hessian via
finite differences, deterministic (doesn't depend on hardware):

| $n$ | calls to `gradient()` | Hessian size |
|---|---|---|
| 5 | 10 (=2n) | 25 floats |
| 20 | 40 | 400 floats |
| 50 | 100 | 2,500 floats |
| 100 | 200 | 10,000 floats |
| 300 | 600 | 90,000 floats — **3.25s** just to assemble, in this engine |

One step of Adam or SGD, at any dimension: **1** call to
`gradient()`, $O(n)$ memory. The Hessian costs $O(n)$ gradient calls
to build (even with exact autodiff, not just finite
differences — it's a structural property, not a limitation of this
implementation), $O(n^2)$ of memory to store, and solving the resulting
linear system costs $O(n^3)$ with direct methods — the same cube of the
condition number that showed up in Module 1 when $X^\top X$ squared
$\kappa$, now appearing as raw computational cost, not
numerical error.

A network with 100 parameters already takes seconds just to assemble the
Hessian in this unoptimized engine. A real model has millions to billions of
parameters — $O(n)$ gradients and $O(n^2)$ of memory, alone, already
make the method infeasible before even considering the $O(n^3)$ of the
linear system. That's why every deep learning optimizer is first-order: it's not
a matter of taste, it's the only class of method that survives at scale.

---

## Closing Module 4

Momentum accelerates but can destabilize; Nesterov fixes the
instability without giving up the acceleration; RMSProp equalizes scale between
parameters, at the cost of losing directional accumulation; Adam combines both,
needing bias correction so as not to distort the first steps;
AdamW fixes an unwanted coupling between regularization and adaptation
that Adam itself introduced without warning. Rosenbrock is a reminder that none
of these methods always wins — the problem's geometry decides. And Newton
shows the other end of the whole spectrum: convergence in few
iterations, at the cost of growth that simply doesn't fit at real
scale. Six trade-offs, each measured with numbers, not assumed by
reputation.
