# 02 — Autodiff

Minimal automatic differentiation engine, two modes: reverse (`Tensor`,
micrograd-style) and forward (`Dual`, dual numbers). Both are validated
independently — against each other, and both against `torch.autograd`
and central finite differences — because no autodiff engine should trust only
itself to prove it's correct.

---

## The reverse engine: three design decisions

**Each operation returns a new `Tensor` that carries a `_backward` closure.**
When `c = a * b`, the resulting `Tensor` stores a function that knows how to
distribute the gradient back to `a` and `b` — the local rule of
multiplication ($\partial c/\partial a = b$, $\partial c/\partial b = a$).
This is what separates autodiff from symbolic differentiation: you never build
an explicit mathematical expression for the derivative, just a graph of local
rules that apply in cascade.

**`backward()` walks the graph in reverse topological order.** It builds the
order via DFS (visiting children before adding itself to the list), then
processes back to front. This guarantees that, when a node's `_backward()`
runs, every node that depends on it has already been processed and has already
deposited its gradient contribution.

**The gradient accumulates with `+=`, never overwrites.** If a variable is used
twice in the graph (e.g. $y = x \cdot x$), it receives a gradient
contribution via **two different paths**, and the multivariable chain
rule says the total derivative is the **sum** of each path's
contributions. Overwriting instead of accumulating is the classic bug that makes
autodiff look like it works on simple examples and fail silently
as soon as a variable is reused.

The three points were validated against `torch.autograd`: a three-variable
expression, a reused variable, and elementwise vector tensors — all four
matched exactly.

## Compositionality: division with no rule of its own

`__truediv__` is implemented as `self * other**-1` — there's no
hand-written `_backward` for division. The correct gradient
emerges automatically from composing two existing rules (`mul` and
`pow`), because the engine doesn't know it's "dividing": it's just chaining
multiplication and exponentiation, and the chain rule takes care of the rest.

This is the central argument for why autodiff scales to networks with
millions of operations: you write the local rule for a handful of primitives
(`+`, `*`, `pow`, `exp`), and any composition of them — no matter how
deep — differentiates automatically, with no new rule needed for each
possible combination.

For the same reason, `x ** Tensor(...)` is explicitly blocked: the engine
knows how to differentiate $x^n$ with respect to $x$ (the power rule), but not with respect to
an exponent that's also a variable — that would require $\partial(x^y)/\partial y
= x^y \ln x$, a primitive that wasn't implemented. It's more honest to block it
with a clear message than to let someone discover it via a
silently wrong gradient.

---

## Central finite differences: the oracle of oracles

`numerical_gradient` serves as an independent validation for any other
autodiff engine — it doesn't depend on torch existing, only on the
definition of the derivative:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}
$$

The truncation error of this approximation is $O(h^2)$ — so "decreasing $h$"
always seems to improve precision. It doesn't. Measured empirically on a
function with a known closed-form derivative ($f(x)=\sum x^3$):

| $h$ | relative error |
|---|---|
| $10^{-1}$ | $6.8\times10^{-3}$ |
| $10^{-3}$ | $6.8\times10^{-7}$ |
| $10^{-5}$ | $8.5\times10^{-11}$ ← optimal |
| $10^{-8}$ | $9.8\times10^{-9}$ |
| $10^{-12}$ | $8.9\times10^{-5}$ |

The error drops, reaches a minimum near $10^{-5}$–$10^{-6}$, and rises again
as $h$ keeps shrinking. Near machine precision
($\varepsilon \approx 2.2\times10^{-16}$), the subtraction
$f(x{+}h) - f(x{-}h)$ becomes dominated by rounding noise —
catastrophic cancellation, the same numerical disease as Gram-Schmidt in
Module 1, only here hitting the very definition of the derivative, not a
side effect of an algorithm. The theoretical optimum sits near
$\varepsilon^{1/3}$ (not $\varepsilon^{1/2}$, which would be the naive
intuition) — worth revisiting Phase 1.1 (floating-point error) to
understand why.

---

## Jacobian, reverse mode: m passes, one per output

For $f: \mathbb{R}^n \to \mathbb{R}^m$, each row of the Jacobian comes from an
independent `backward()`, seeding the corresponding output with gradient
1 and the rest with 0. This costs **$m$ full passes** — rebuilding
the graph from scratch each row, because this engine doesn't retain the graph between
backward calls (there's no `retain_graph`, unlike PyTorch). Without retaining,
internal nodes shared between two outputs would accumulate gradient
incorrectly if the same graph were reused without zeroing everything out — it's
exactly this problem that PyTorch's `retain_graph=True` solves,
as an explicit option instead of the default behavior.

The structural point: **reverse mode is optimal when $m \ll n$** — few
outputs, many inputs. That's why training neural networks uses reverse
mode: the loss is scalar ($m=1$), the parameters number in the millions ($n$ large), and
a single `backward()` gets the entire gradient.

---

## Forward mode: dual numbers

A dual number $x + \varepsilon x'$, with $\varepsilon^2 = 0$, carries
value and derivative together, propagated in a single forward pass. There's no
graph, no backward phase — the derivative already comes out ready in the dual
part, in the same pass that computes the value. The product rule falls out for
free from the algebra: $(a+\varepsilon a')(b+\varepsilon b') = ab + \varepsilon(ab'+a'b)$,
since $\varepsilon^2$ discards the cross term.

`jacobian_forward` is the exact mirror of the reverse version: instead of $m$ passes
(each giving one full row), it takes **$n$ passes** (each giving
one full column — every output, for one input direction).

### The proof, with exact call counts

Two functions, opposite directions of $n$ vs. $m$, instrumented to count
how many times `f` gets evaluated:

| Case | $n$ (inputs) | $m$ (outputs) | forward calls | reverse calls |
|---|---|---|---|---|
| "wide" | 6 | 1 | **6** | **1** |
| "tall" | 2 | 5 | **2** | **5** |

Both modes agree exactly on the result (validated by
`test_forward_and_reverse_modes_agree`) — the only thing that changes is the
cost, and the cost follows $n$ for forward, $m$ for reverse, with no exception.
**There's no universally better mode**: the choice depends entirely on
$f$'s shape. Neural networks have $m=1$ (the loss) and $n$ in the millions —
reverse wins every time, by a huge margin. A system of equations
with few inputs and many outputs would flip the choice.

---

## Closing the module's argument

Five pieces — `Tensor`, compositionality, finite differences,
reverse Jacobian, dual numbers — converge on a single point: **the "right"
mode of autodiff doesn't exist in isolation from the shape of the problem.** Reverse
wins when the output is scalar and the inputs are many (the case for
training networks); forward wins in the opposite case. And finite
differences — which is neither forward nor reverse, it's just the definition of
the derivative applied naively — remains indispensable not because it's fast, but
because it's the only one of the three that can't be wrong in the same way as the
other two: any bug in a `_backward` rule or in `Dual` gives itself
away against it.

---

## Hessian: a deliberate hybrid, not second-order autodiff

This engine doesn't do "real" second-order automatic differentiation.
That would require the `backward()` pass itself to be part of a
differentiable graph — every operation inside `_backward()` would have to
be built with `Tensor`s, not raw `numpy` arithmetic, so that a
second `backward()` could propagate through the first pass. That's
exactly what PyTorch does with `create_graph=True`, and it's a real
extension of the engine, not a small tweak — each of the `_backward`
closures would have to become, itself, a differentiable composition
of `Tensor`s.

Instead, `hessian()` is a deliberate hybrid:

$$
H_{:,j} \approx \frac{\nabla f(x_0 + h\,e_j) - \nabla f(x_0 - h\,e_j)}{2h}
$$

The gradient $\nabla f$ comes from `gradient()` — **exact**, via `Tensor.backward()`,
with no truncation error. The central finite difference is applied
**on top of that already-exact gradient**, not directly on $f$. The result:
half the Hessian comes from autodiff (the part each column represents —
the entire gradient vector, at each perturbed point), and the other half
comes from finite differences (how the columns combine). It's
cheaper than full second-order autodiff, and more precise than applying
finite differences twice in cascade (which would double the accumulated
truncation error $O(h^2)$).

## The test that would have caught a real bug: Schwarz's symmetry

`test_hessian_is_symmetric` isn't an incidental check — it encodes
**Schwarz's theorem** (mixed partial derivatives commute,
$\partial^2f/\partial x_i\partial x_j = \partial^2f/\partial x_j\partial x_i$,
for sufficiently smooth $f$). In a real double-backward Hessian
implementation, that symmetry would come guaranteed by the computational
graph's own structure. Here, **it isn't guaranteed by construction**
— `hessian()` computes each column independently, perturbing one
variable at a time, with no logic that forces $H_{ij} = H_{ji}$
explicitly.

The observed symmetry (~$10^{-11}$ of residual asymmetry, just floating-point
noise) is an **emergent** property of two independent numerical
processes — exact gradient plus finite difference —
agreeing because the underlying math is symmetric, not because the
code guarantees it. If this test ever failed, it would signal a real
implementation bug (e.g. an index error swapping $i$ and $j$ somewhere),
not expected numerical imprecision — it's exactly the kind of test
worth keeping even after it "obviously looks like it will pass".

## Closing Phase 1.1's vector calculus

Gradient, Jacobian, Hessian — the three derivatives Phase 1.1 asked for
("gradients, Jacobians, Hessians, chain rule") now have their own
implementation, validated by three independent oracles that never
agree by accident: `torch.autograd` (an entire other autodiff engine),
central finite differences (the definition of the derivative, no engine at
all), and — in the Hessian's case — Schwarz's theorem (a
mathematical property the code doesn't impose, only inherits).

---

## The capstone: logistic regression, from scratch, against sklearn

The engine's last piece: `sum()`, the reduction that was missing to compute the
dot product between a weight vector and a feature vector —
$z = \sum_i w_i x_i + b$ — without it, there was no way to collapse eight
contributions into a single scalar output. Like every piece in this module, the
gradient of `sum()` is simple and mechanical: $\partial(\sum_i x_i)/\partial
x_i = 1$ for every $i$, the output gradient spreads back equally to
every element that went into the sum.

With `sum()`, `log()`, `exp()`, and `SGD`, the entire logistic regression —
sigmoid, cross-entropy loss, training loop — is built by
composition, with no new primitive. `sigmoid(z) = 1/(1+exp(-z))` and the
loss use only what already existed. This is the module's compositionality
argument carried through to the end: six primitives (`+`, `*`, `pow`, `exp`,
`log`, `sum`) are enough to train a real classifier.

### The test that proves the engine works, not just that it compiles

Trained on Module 0's real HVI data — 180 training bales, a synthetic
"premium" label via a linear combination of strength, uniformity,
length, and trash plus Gaussian noise (a linear threshold with
real class overlap, not a trivially separable problem) —
the model matched `sklearn.linear_model.LogisticRegression`'s test
accuracy **exactly** on the same split (0.7143 for both), and the
learned weights came out close, coefficient by coefficient:

| Feature | Our weight | sklearn's weight |
|---|---|---|
| strength | 0.779 | 0.895 |
| uniformity | 0.647 | 0.699 |
| uhml | 0.633 | 0.608 |
| trash | -0.665 | -0.752 |

The small differences come from the optimizer: our plain, full-batch SGD, 120
steps; `sklearn` uses L-BFGS with light L2 regularization by default — two
different optimization paths converging near the same minimum,
because the problem is well-conditioned enough for that to matter little.

This closes the whole module's central argument: a ~200-line engine,
validated piece by piece against `torch`, central finite differences, and
mathematical properties (Schwarz, the Rayleigh theorem), composes a system
that trains a real model and generalizes as well as a production
library — not by coincidence, but because each piece was proven correct
before composing the next.
