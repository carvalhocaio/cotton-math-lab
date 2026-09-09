"""Maximum likelihood estimation (MLE)."""

import numpy as np


def mle_normal(data: np.ndarray) -> tuple[float, float]:
    """MLE of μ and σ from a sample ~ Normal(μ, σ²).

    Maximizing a Normal's log-likelihood has a closed-form solution:
    μ_MLE is the sample mean — always unbiased, for any n. σ²_MLE is the
    sample variance divided by n, NOT by n-1 (the "Bessel correction"
    used in almost every other statistical context). This choice isn't
    arbitrary: it falls straight out of zeroing the log-likelihood's
    derivative with respect to σ², with no post-hoc bias adjustment —
    and that's exactly why σ²_MLE underestimates the population variance
    by the exact ratio (n-1)/n.
    """
    mu_hat = float(np.mean(data))
    sigma_hat = float(np.sqrt(np.mean((data - mu_hat) ** 2)))
    return mu_hat, sigma_hat
