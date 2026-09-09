"""Confidence interval (frequentist) vs. credible interval (Bayesian) -
the same apparent question, two different mathematical guarantees."""

import numpy as np
from scipy import stats


def wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson confidence interval for a binomial proportion.

    Uses no prior — it's the inversion of a hypothesis test: the set of
    values p₀ for which a score test (H₀: p=p₀) would not be rejected at
    the given significance level. The frequentist guarantee comes
    entirely from this construction, and doesn't depend on any prior
    belief about p.
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return center - margin, center + margin


def beta_credible_interval(
    alpha: float, beta: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Credible interval from the quantiles of a Beta posterior.

    Unlike the confidence interval: here p IS treated as a random
    variable (given the posterior), and the interval is literally "95%
    posterior probability that p is here" — a direct statement about the
    parameter, conditioned on the observed data AND the chosen prior.
    That extra condition ("AND the prior") is what the common incorrect
    reading of confidence intervals tends to forget.
    """
    tail = (1 - confidence) / 2
    return stats.beta.ppf(tail, alpha, beta), stats.beta.ppf(1 - tail, alpha, beta)
