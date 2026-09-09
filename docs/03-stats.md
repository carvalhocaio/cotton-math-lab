# 03 — Probability and Statistics

Estimators applied to the micronaire of a batch of bales (one
farm/season): MLE as the starting point, MAP as the counterpoint that
shows where a prior genuinely helps. Each estimator is validated by at
least two independent routes — never just against one library.

---

## MLE: closed form, validated twice

For data $\sim \mathcal{N}(\mu, \sigma^2)$, maximizing the log-likelihood
has a closed-form solution:

$$
\hat\mu = \bar{x}, \qquad \hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2
$$

Two independent validations, neither of them "trust the formula":

**Against `scipy.stats.norm.fit`** — matched exactly, difference $0.0$.

**Against Module 2's own autodiff engine** — maximizing the
log-likelihood numerically via `SGD`, starting from a deliberately bad
guess ($\mu_0=0$, far from the real value $\approx4.3$), with
$\sigma$ parameterized as $\exp(\text{log\_sigma})$ to guarantee
positivity without needing constrained optimization. It converged to the same
point as the closed form, difference $\sim10^{-9}$. This second validation
matters more than it seems: it proves, at the same time, that the closed
form is correct **and** that Module 2's optimization engine generalizes to
a problem that isn't any kind of neural network — it's just another loss
surface.

### The detail almost everyone gets wrong: $\hat\sigma^2$ is biased

$\hat\mu$ is always unbiased, whatever $n$ is. $\hat\sigma^2$
**isn't** — it underestimates the population variance by the exact ratio
$\frac{n-1}{n}$, measured via Monte Carlo (3000 samples of size $n=8$):
observed ratio $\approx0.857$, expected $0.875$, within noise.

This is the exact origin of the "Bessel correction" ($/(n-1)$ instead of $/n$) that
shows up in every sample variance calculator, `np.var(ddof=1)`
included: it exists specifically to undo the bias MLE
introduces. MLE isn't wrong out of carelessness — it maximizes the observed
data's likelihood, and a sample always varies less around its *own*
mean than around the true (unknown) population mean. The
`ddof=1` is a post-hoc correction that MLE, by definition, doesn't apply.

---

## MAP: the prior as a regularizer that knows when to stop

For $\mu$ with known $\sigma$ and a conjugate prior
$\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$, the posterior is also Normal, and its
mean (= mode, by symmetry) is a **precision**-weighted average
(inverse of the variance) between the prior and the data:

$$
\mu_{\text{MAP}} = \frac{\mu_0/\tau_0^2 + n\bar{x}/\sigma^2}{1/\tau_0^2 + n/\sigma^2}
$$

Validated against a grid search on the real posterior (prior × likelihood,
maximized point by point on a grid of 20 thousand $\mu$ values) — difference
$3\times10^{-6}$, limited only by the grid's resolution.

### The two limits, confirmed

Nearly non-informative prior ($\tau_0^2 \to \infty$): $\mu_{\text{MAP}} \to
\bar{x}$, the MLE — matched with difference $<10^{-4}$. Extremely
confident prior ($\tau_0^2 \to 0$): $\mu_{\text{MAP}} \to \mu_0$, ignoring the
data entirely — same precision.

### The number that proves the trade-off promised since the start of the lab

With $n=4$ and a prior that's only **approximately** right ($\mu_0=4.2$, truth
$=4.3$ — not a prior "cheating" with the exact answer), Monte Carlo with
5000 repetitions:

| $n$ | MSE(MLE) | MSE(MAP) | reduction |
|---|---|---|---|
| 4 | 0.0392 | 0.0197 | **≈50%** |
| 200 | 0.00080 | 0.00079 | ≈2%, within noise |

MAP's advantage doesn't come from "cheating" by knowing the answer — it comes
from combining two sources of information (prior + data), and with little data the
extra source carries a lot of weight. As $n$ grows, the data
drown out the prior on their own, and the advantage disappears — exactly as it
should: MAP isn't "better than MLE" in general, it's better
**specifically in the scarce-data regime where MLE has high variance**.

### The connection worth an entire career

A Normal prior centered at zero over a neural network's weights **is**
L2 regularization — the Bayesian derivation of MAP with that specific prior
produces exactly the $\lambda\|\theta\|^2$ term that shows up in every
regularized loss function. What in Module 4 will look like an "engineering
trick" (adding a penalty term to prevent overfitting) is, seen from
this angle, the same math we just validated here:
an informative prior reducing variance at the cost of a bit of bias,
more useful the less data there is.

---

## Beta-Binomial: the natural counterpart to Normal-Normal, for binary data

Same logic as the normal MAP — combining prior and data —, but for
a proportion instead of a continuous mean. Prior $p \sim \text{Beta}(\alpha,
\beta)$, observation: $k$ bales out of specification among $n$ inspected,
$\sim \text{Binomial}(n, p)$. The posterior is exact, with no
approximation:

$$
p \mid k, n \;\sim\; \text{Beta}(\alpha + k,\; \beta + n - k)
$$

Unlike the previous cycle's Normal-Normal — which is only exact if $\sigma$
is known —, Beta-Binomial is conjugate **unconditionally**: there's no
extra parameter that needs to be assumed fixed for the conjugacy
to work.

### Validation

Against a grid search (prior × binomial likelihood, integrated
numerically over a grid of 50 thousand points): difference $0.0$ in the
tested scenario (6 out of spec among 40 inspected). Against `scipy.stats.beta`:
exact. And the usual two limits confirm: a nearly-flat prior
($\alpha,\beta \to 0$) makes the posterior mean converge to the MLE $k/n$; a
concentrated prior ($\alpha+\beta$ large) with little observed data
($n=3$) pulls the posterior mean back close to the prior's mean,
resisting a small, noisy sample.

### The property that matters more than the formula: sequential updating

Updating the posterior in two batches — first with 4 successes in 20
trials, then with 2 in 20 — gives **exactly** the same posterior as
updating all at once with the combined 6 in 40. It's not
approximately equal: the parameters match bit for bit.

This isn't an implementation curiosity — it's the property that makes
Bayesian inference genuinely **incremental**. Last week's posterior
*is* this week's prior. You can update your belief about the
proportion of out-of-spec bales with every new inspected batch,
without ever needing to reprocess the entire history, and the final result
doesn't depend on having processed everything at once or in fifty pieces
over fifty weeks. A pure MLE system, recomputed from scratch every
batch, doesn't get this property for free — each recomputation is
independent of the previous one, and "remembering" requires keeping the entire
raw history, not just two numbers ($\alpha$, $\beta$).

### Connecting the module's two conjugates

Normal-Normal and Beta-Binomial solve the same structural problem —
combining a prior belief with new evidence, weighted by each one's relative
confidence — in two different domains: one continuous (the mean of
a normal distribution), the other discrete (a proportion). How they combine
differs (precision-weighted average vs. sum of counts), but
the logic is identical, and it's the same logic that shows up again, disguised as
"regularization" or "Laplace smoothing", in practically every corner
of machine learning.

---

## The module's throughline so far

MLE answers "what parameter makes the observed data most likely,
with no other information". MAP answers the same question while admitting
that you already knew something before looking at the data. Neither is
"more correct" — they're answers to different questions, and the right
choice depends on how much data you have and how much you trust what you already
knew.

---

## Bootstrap: when there's no conjugate at hand

The previous two cycles relied on exact conjugacy — Normal-Normal,
Beta-Binomial. But most statistics of interest have no conjugate
at all: the median, correlation, a ratio of variances, a regression
coefficient. Bootstrap solves this without assuming anything about the
statistic's form.

The whole idea in one sentence: resample the observed data **with
replacement**, many times, recompute the statistic of interest on each
resample, and use the percentiles of that simulated distribution as the interval.
There's no standard-error formula at all — the sampling
variability is estimated empirically, treating the observed sample as if it were
the population itself.

### Validation on two fronts

Against `scipy.stats.bootstrap`, with the same seed: matches essentially
exactly, both for the mean and for the **median of an
exponential distribution** (skewed on purpose — no simple closed-form
standard-error formula for the median, exactly the case where bootstrap matters).

More rigorous: empirical coverage measured by repeating the entire
experiment. A 95% bootstrap CI, repeated 1000-2000 times with fresh
samples, captured the truth **93-94%** of the time — not exactly 95%.
That's not a bug: the percentile method has a known under-coverage bias,
more pronounced in modest samples (there is a fix — *bias-corrected
and accelerated*, BCa — out of scope for this cycle). The number matters more
than the intuition: "it seems like it should give 95%" isn't the same as measuring that it does.

---

## Confidence interval vs. credible interval: the most common confusion in statistics

Two constructions that answer **different** questions, even when
the output number looks like the same kind of thing (an interval, a 95%
confidence level).

**Confidence interval (Wilson, for a proportion)** uses no
prior — it's the inversion of a hypothesis test: the set of values $p_0$
for which a score test wouldn't be rejected. The guarantee is about the
**procedure**: repeating the entire experiment many times, 95% of the
constructed intervals will contain the true $p$. For a single already-computed interval,
$p$ either is or isn't inside it — there's no "95% chance"
about that specific interval, because in the frequentist framing $p$ is
fixed, not random.

**Credible interval** comes from the quantiles of the Beta posterior. Here $p$
**is** treated as a random variable (given the observed data), and the
statement is direct: "95% posterior probability that $p$ is here,
given this data and this prior". The extra condition — "and this prior" — is
exactly what the popular, incorrect reading of confidence intervals tends to forget.

### The proof, with measured numbers

A single concrete example already shows the divergence: with $k{=}6$ out of
specification among $n{=}40$ inspected, and a strong prior centered on 0.20
(fixed **before** seeing the data):

| | lower bound | upper bound |
|---|---|---|
| CI (Wilson, 95%) | 0.071 | 0.291 |
| Credible (strong prior) | 0.145 | 0.244 |
| Credible (weak prior) | 0.071 | 0.292 |

The CI never changes, because it uses no prior at all. The credible interval with a
strong prior is noticeably narrower and shifted — the prior belief carried
weight. The credible interval with a weak prior practically coincides with the CI:
**when the prior doesn't carry weight, the two framings converge
numerically**, even while remaining philosophically distinct.

The part that really separates the two is **coverage under repetition**, with
the truth fixed at $p{=}0.15$ and a **mismatched** prior
(centered on 0.20, fixed before any data):

| Method | Measured coverage (nominal: 95%) |
|---|---|
| CI (Wilson) | 96.3% |
| Credible, mismatched prior | **74.1%** |
| Credible, weak prior | 96.3% |

The CI keeps its 95% guarantee **regardless** — it's a property of
the construction, it doesn't depend on anything being "right" about $p$. The credible
interval with a wrong prior does **not** keep that guarantee: it fell 21
percentage points below nominal. This doesn't invalidate the
Bayesian reading — the interval remains correct as a statement of posterior
belief, given that specific prior. What's proven is that this
belief statement **isn't the same thing** as the frequentist guarantee of
repeated coverage, and treating one as if it were the other is exactly the
most common mistake in how statistics gets communicated.

### The rule of thumb for deciding which to use

Neither is "more correct" — they answer different questions. Use a CI
when the question is "what guarantee does this procedure have, under
repetition, without me needing to assume anything prior about the parameter". Use a
credible interval when you **have** a genuine prior belief (history from
previous seasons, domain knowledge about the gin) and want
a direct probability statement about the parameter, conditioned
on that belief — but then the statement's validity depends entirely on
how reasonable the chosen prior was, something the method itself will never
warn you about if you got it wrong.

---

## Closing Module 3

MLE answers "what the data alone suggests". MAP and Beta-Binomial
answer the same question while admitting a prior belief, and show
(with numbers, not just theory) that this belief helps most exactly when
the data is scarce. Bootstrap does without a closed formula when no
conjugate exists. And confidence vs. credible closes the module with the most
applicable point of all: two numbers that look like the same thing can carry
completely different mathematical guarantees — and the difference only shows up
when someone measures it, not when someone assumes it.
