"""Conjugate Beta-Binomial: proportion of bales out of specification."""


def beta_binomial_posterior(
    k: int, n: int, alpha_prior: float, beta_prior: float
) -> tuple[float, float]:
    """Posterior parameters, given prior p ~ Beta(α, β) and k successes
    in n trials ~ Binomial(n, p).

    Exact conjugation: the posterior is Beta(α+k, β+n-k) — the same
    family as the prior, just with the parameters shifted by the
    observed data. There's no approximation here at all, unlike the
    Normal-Normal from the previous cycle (which is also exact, but
    requires a known σ); Beta-Binomial is unconditionally exact.
    """
    return alpha_prior + k, beta_prior + (n - k)


def posterior_mean(alpha: float, beta: float) -> float:
    """Mean of the Beta(α, β) distribution — the Bayesian point estimate."""
    return alpha / (alpha + beta)
