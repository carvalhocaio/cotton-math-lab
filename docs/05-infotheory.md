# 05 — Information Theory

Entropy, cross-entropy, and KL first in discrete form (distributions
over categories — bale quality grades), then continuous (KL
between densities). The module's central cycle is KL's asymmetry — the
reason "distance" is the wrong word to describe what it measures.

---

## Entropy, cross-entropy, KL: one identity, not three separate concepts

$$
H(p) = -\sum_x p(x)\log p(x), \qquad H(p,q) = -\sum_x p(x)\log q(x)
$$

$$
D_{KL}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p)
$$

The third formula isn't a coincidence — it's an identity, tested
directly: `H(p,q) = H(p) + D_KL(p‖q)`, matches exactly. Cross-entropy is the
average cost of encoding samples from $p$ using a code optimized for
$q$; entropy is the lowest possible cost (a code optimized for $p$
itself); KL is the difference — how much extra you pay for using the wrong
code.

### Two properties that support everything else in the module

**Gibbs' inequality**: $D_{KL}(p\|q) \geq 0$ always, for any pair
of distributions — tested on 5000 random pairs via the simplex
(Dirichlet), zero violations. Equality only when $p=q$ exactly.

**The uniform maximizes entropy**: among 3000 random distributions
over 5 categories, none exceeded $H(\text{uniform}) = \log 5$. Makes
sense: entropy measures "how much uncertainty is left", and no distribution
has more uncertainty over $n$ outcomes than giving equal weight to all of them.

---

## KL's asymmetry: why "distance" is the wrong word

$D_{KL}(p\|q) \neq D_{KL}(q\|p)$ in general — KL isn't symmetric, so it isn't
a distance metric in the formal sense (it satisfies neither the triangle
inequality nor symmetry). This asymmetry isn't just a technical detail:
it produces **qualitatively different** behaviors when used as an
optimization objective.

### The experiment: fitting a Gaussian to a bimodal target

Target: $p(x) = 0.5\,\mathcal{N}(x;-3,1) + 0.5\,\mathcal{N}(x;3,1)$ — two
well-separated modes. Fitting a Gaussian $q$ by minimizing each direction of KL.

**Forward KL** ($D_{KL}(p\|q)$, minimized over $q$): has a closed form
for $q$ within the exponential family — moment matching, $\mu = E_p[x]$,
$\sigma^2 = \text{Var}_p[x]$. Result: $\mu=0$, $\sigma=3.16$ — a
**wide** Gaussian, covering both modes (and wasting mass in
the valley between them, where $p$ is low). This moment-projection
property holds for any exponential family, not just the Gaussian — it's
closed-form math, not an experimental coincidence.

**Reverse KL** ($D_{KL}(q\|p)$, minimized over $q$): no general closed form,
requires numerical optimization — and the result **depends on where the
optimization starts**:

| Start | $\mu^*$ | $\sigma^*$ | final KL |
|---|---|---|---|
| near the left mode | -2.984 | 1.023 | 0.689 |
| near the right mode | 2.984 | 1.023 | 0.689 |
| in the middle (between modes) | 0.000 | 2.777 | 0.834 (worse!) |

Starting near either mode, $q$ **collapses** onto it — mean and standard deviation
practically replicate one component of the mixture, ignoring the other
entirely. Starting in the middle, the optimization gets stuck at a
worse local minimum. This isn't an optimization failure: it's the geometry of
the reverse KL surface, which has a local minimum near each mode of $p$, because $q$
is penalized for placing mass where $p$ is low — but is never required
to cover all of $p$'s mass.

### Mean-seeking vs. mode-seeking

The compact takeaway worth carrying around: **forward KL "spreads out"** (covers
the entire support of $p$, accepting wasted mass in low-density
regions, because the term $p(x)\log(p(x)/q(x))$ penalizes heavily when $p(x)>0$ but
$q(x)\to0$). **Reverse KL "collapses"** (concentrates on a single mode, because
the term $q(x)\log(q(x)/p(x))$ is only evaluated where $q$ places mass —
placing mass in a low-density valley of $p$ is already expensive enough for
$q$ to prefer avoiding it entirely).

### Why this matters for the rest of your career

This distinction isn't theoretical trivia — it's the structural reason behind
design decisions in modern ML:

- **Knowledge distillation** typically minimizes forward KL (the
  teacher's output is $p$, the student's is $q$) — it wants the student to cover
  the teacher's entire behavior, even in rare cases.
- **DPO and RLHF** frequently involve reverse KL between the optimized
  policy and a reference policy — the goal is to stay close to a
  specific behavior, not cover the entire distribution of
  possibilities, and mode collapse is, in that context, a feature
  (focus), not a bug.
- **VAEs** use reverse KL to regularize the latent space
  precisely because mode collapse, there, produces more
  "decodable" representations — covering the entire space of possibilities would
  produce a posterior too diffuse to be useful.

Understanding which KL direction a method uses — and why — is what separates
"I know there's a loss function called KL" from "I know how to predict what
behavior this specific choice will produce".

---

## Mutual information: what correlation doesn't see

$$
I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

Estimated via binning: discretizes two continuous variables into
bins, computes the joint histogram, applies the formula above over the
empirical probabilities. Validated against `sklearn.metrics.mutual_info_score`
using the same bin edges in both computations — difference
$2\times10^{-16}$, machine precision.

### The experiment that proves the tool's value

$Y = X^2$, with $X$ uniform in $[-1,1]$ (symmetric around zero):

| Measure | Value |
|---|---|
| Pearson correlation | 0.006 (essentially zero) |
| Mutual information | 1.96 (strong dependence) |

The dependence between $X$ and $Y$ is **perfect and deterministic** —
knowing $X$, you know $Y$ exactly. But it's symmetric around zero, so
the covariance (Pearson's numerator) cancels out: positive and
negative values of $X$ contribute with opposite signs, and the sum comes out
approximately zero. Pearson literally can't see this
dependence, because it's only sensitive to the **linear** component of any
relationship. Mutual information has no such blind spot — it measures
statistical dependence of any form, linear or not.

### Application to real HVI data

Comparing a pair of features correlated by construction in Module 0's
generator (`uhml`, `uniformity`, correlation 0.55) against an
independent-by-construction pair (`micronaire`, `rd`, correlation 0):

$$
\text{MI(uhml, uniformity)} = 0.208 \qquad \text{MI(micronaire, rd)} = 0.026
$$

Almost 8× larger — mutual information recovered the real dependence
structure the generator planted, with no linearity assumption. For
feature selection on real HVI data, where relationships between
physical parameters are rarely purely linear (fiber maturity
affects micronaire and strength in ways that aren't necessarily
additive), this is a concrete advantage over looking only at the
correlation matrix.

---

## JS as a drift detector: the application that anticipates production monitoring

KL has two problems for serving as "how different is this week's season
from the baseline season": it isn't symmetric (KL(new_season‖baseline) ≠
KL(baseline‖new_season) — which one is "the right answer"?), and it can
blow up to infinity if the new season has values the baseline
distribution never observed. Jensen-Shannon solves both:

$$
D_{JS}(p,q) = \tfrac{1}{2}D_{KL}(p\|m) + \tfrac{1}{2}D_{KL}(q\|m), \qquad m = \tfrac{p+q}{2}
$$

Symmetric by construction (tested exactly), bounded between 0 and $\log 2$
(tested at both extremes — identical distributions give 0, distributions
with completely disjoint support give $\log 2$ exactly). Validated against
`scipy.spatial.distance.jensenshannon` — which returns the *square root* of
the divergence (the JS distance), so the comparison squares it first.

### The proof that it works as a detector, not just as a metric

Three seasons simulated via Module 0's generator: a baseline, a second
sample from the **same** distribution (just sampling noise — different
seeds, no real change), and a third with a real shift of
0.6 in the micronaire mean (simulating a genuinely different season).

| Comparison | $D_{JS}$ |
|---|---|
| Baseline vs. same distribution (sampling noise) | 0.0033 |
| Baseline vs. shifted distribution (real drift) | **0.219** |

Nearly 65× larger in the real-drift case. `detect_drift()` uses exactly
this separation: a threshold (calibratable from historical data on
how much genuine sampling noise typically produces) decides whether the
observed divergence is too large to be just natural variation between
similar seasons.

This is, literally, the central technique for monitoring models in
production — detecting when the distribution of input data has changed
enough to cast doubt on whether a model trained in the past is still
reliable. Phase 3 of the roadmap (evaluation and MLSecOps) will encounter this
exact tool again, only applied to *features of a model in production*
instead of *HVI parameters of a season* — the method doesn't change, only the domain.

---

# Closing Phase 1.1: Mathematical Foundations

Five modules, one goal: fluency that survives outside the context in
which it was learned.

**Linear Algebra** — three eigenvalue methods, each trading one pain
for another (speed, precision, or both); PCA revealing that the "obvious"
way to decompose data (via covariance) loses precision exactly
when it most needs to work, and direct SVD on the data matrix solving
that at the cost of never squaring the condition number.

**Calculus / Autodiff** — a ~300-line engine that differentiates any
composition of six primitives, the two modes (reverse, forward) trading
$m$'s cost for $n$'s cost depending on the problem's shape, and a
real classifier trained from scratch that matched `sklearn`'s accuracy.

**Probability and Statistics** — MLE getting it wrong in a predictable
way (variance biased by $(n-1)/n$), MAP fixing it exactly when the
data is scarce, and the phase's most applicable proof: a confidence
interval and a credible interval aren't the same guarantee, and the
difference costs 21 percentage points of coverage when confused.

**Optimization** — six optimizers validated trajectory by trajectory
against `torch`, Rosenbrock debunking the "Adam is always better" cliché, and
Newton showing the price of fast convergence: $O(n)$ gradients and
$O(n^2)$ of memory just to assemble the Hessian, before even the $O(n^3)$ of
solving the system — the structural reason every production deep learning
optimizer is first-order.

**Information Theory** — entropy, KL, and the asymmetry that separates
"covering the entire distribution" from "collapsing onto a mode", with
direct consequences for how distillation and DPO behave; mutual
information seeing dependence that Pearson doesn't; and JS closing
with a real monitoring application, anticipating Phase 3.

The pattern running through all five modules, not by coincidence: **the
most direct way to solve a numerical problem is rarely the most
stable one, and the difference only shows up when someone measures it** —
squared condition numbers, catastrophic cancellation, estimator bias,
optimizer instability, mode collapse. None of these trade-offs was accepted on
textbook authority in this lab; all were reproduced, measured, and only
then documented. It's this disposition — distrusting the obvious
answer until it survives a test — that Phase 1 set out to
train, and it's exactly what separates reading a paper from judging it.
