# 00 — Synthetic HVI Data Generator

## Why synthetic data, and not real company data?

Using synthetic data offers an important pedagogical advantage: every model parameter is known by construction. Since the values used to generate the data are defined ahead of time, it's possible to check whether the implemented algorithms can recover those same parameters, objectively evaluating their correctness.

With real data, the true parameters are unknown. So even if an estimate looks plausible, there's no exact reference against which to measure its precision. With synthetic data, on the other hand, the estimated value can be compared directly against the value used to generate the data, allowing both the implementation and the statistical behavior of the method to be validated.

This strategy becomes especially important in Module 3, where Maximum Likelihood Estimation (MLE) must recover the parameters used to generate the distribution. If the estimates converge to the known values, that's concrete evidence that the implementation is correct before applying it to real data.

---

## Why Cholesky, and not `rng.multivariate_normal`?

Fact: if $z \sim \mathcal{N}(0, I)$ and $\Sigma = LL^\top$ (Cholesky), then

$$
\mu + Lz \sim \mathcal{N}(\mu, \Sigma).
$$

Although the `rng.multivariate_normal` function produces the same result, explicitly implementing the Cholesky factorization makes the data generation process transparent. Instead of using a high-level function as a "black box", it becomes clear that generating correlated variables consists of transforming a standard normal distribution via the matrix $L$, which acts as a "square root" of the covariance matrix.

Beyond the pedagogical gain, this factorization reappears in several algorithms studied later. In Module 4, it's used to build preconditioners that improve the stability and speed of numerical methods. In Module 5, the Cholesky decomposition is used in calculations involving Gaussian distributions, such as the Kullback-Leibler (KL) divergence, where operations with determinants and inverses of covariance matrices are needed.

So, manually implementing the transformation helps build an understanding of a concept that will be reused throughout the entire lab.

---

## The Fisher transform (variance-stabilizing transform)

Fact: the Pearson coefficient lives in $[-1,1]$ and its sample variance **depends on the value itself**. The Fisher transform stabilizes that variance:

$$
z = \operatorname{arctanh}(r)
= \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right),
\qquad
\operatorname{SE}(z)\approx\frac{1}{\sqrt{n-3}}.
$$

Using an absolute tolerance, such as $|r_{\text{emp}} - r_{\text{true}}| < 0{.}02$, isn't a good strategy because the variability of the estimate depends on the correlation value itself. Correlations close to $\pm1$ naturally have lower variance than correlations near zero. So the same absolute difference can represent an expected error in one case and a statistically improbable error in another.

The Fisher transform solves this problem by converting the correlation to a scale where the variance is approximately constant, with a standard error that depends only on the sample size. This allows estimates to be compared using the same statistical criterion, regardless of the true correlation value.

This principle shows up in several other contexts in statistics. The logit transforms probabilities bounded to the $(0,1)$ interval into an unbounded scale, while the logarithm transforms strictly positive variables to reduce skew and stabilize variance. Likewise, the link functions of Generalized Linear Models (GLMs) use transformations to make modeling better suited to the data's statistical properties.

---

## Recorded design decisions

- **Returns an `ndarray (n, k)` instead of a `DataFrame`:** keeps the function independent of analysis libraries, reduces overhead, and makes integration with numerical algorithms that operate directly on NumPy arrays easier.

- **Explicit `seed` instead of global state:** guarantees reproducibility of experiments and prevents different parts of the code from interfering with each other by sharing the same random number generator.

- **`eq=False` on the `frozen dataclass`:** prevents automatic comparisons based on array equality, avoiding ambiguities and errors like *"The truth value of an array is ambiguous"*.

- **Order of validations (dimension → positivity → symmetry → diagonal → positive-definite):** checks simple, cheap conditions first before running more computationally expensive tests, producing clearer and more specific error messages.

- **Error message reports the smallest eigenvalue:** besides indicating that the matrix isn't positive-definite, it shows by how much it violates that condition, making both debugging and adjusting the parameters used easier.
