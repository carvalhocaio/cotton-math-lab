"""Maximum a posteriori (MAP) estimation, Normal-Normal conjugate."""

import numpy as np


def map_normal_mean(
    data: np.ndarray,
    sigma: float,
    prior_mean: float,
    prior_std: float,
) -> float:
    """MAP of μ for data ~ Normal(μ, σ²) with σ KNOWN, prior
    μ ~ Normal(prior_mean, prior_std²).

    Normal-Normal conjugate: the posterior of μ is also Normal, and its
    mean (which coincides with the mode — the posterior is symmetric) is
    a PRECISION-weighted average (inverse variance) between the prior
    and the data:

        μ_MAP = (prior_mean·τ₀ + n·x̄·τ_data) / (τ₀ + n·τ_data)

    where τ₀ = 1/prior_std² and τ_data = n/σ². As n grows, the data's
    weight dominates and μ_MAP → x̄ (the MLE). As prior_std → 0, μ_MAP →
    prior_mean, no matter how much data exists — an overly confident
    prior can dominate even a large sample.
    """
    n = len(data)
    sample_mean = float(np.mean(data))

    prior_precision = 1.0 / prior_std**2
    data_precision = n / sigma**2

    return (prior_mean * prior_precision + sample_mean * data_precision) / (
        prior_precision + data_precision
    )
