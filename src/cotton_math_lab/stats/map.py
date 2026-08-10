"""Estimação máximo a posteriori (MAP), conjugado Normal-Normal."""

import numpy as np


def map_normal_mean(
    data: np.ndarray,
    sigma: float,
    prior_mean: float,
    prior_std: float,
) -> float:
    """MAP de μ para dados ~ Normal(μ, σ²) com σ CONHECIDO, prior μ ~ Normal(prior_mean, prior_std²).

    Conjugado Normal-Normal: a posterior de μ também é Normal, e sua média
    (que coincide com a moda — a posterior é simétrica) é uma média
    ponderada por PRECISÃO (inverso da variância) entre prior e dados:

        μ_MAP = (prior_mean·τ₀ + n·x̄·τ_dados) / (τ₀ + n·τ_dados)

    onde τ₀ = 1/prior_std² e τ_dados = n/σ². Conforme n cresce, o peso dos
    dados domina e μ_MAP → x̄ (o MLE). Conforme prior_std → 0, μ_MAP →
    prior_mean, não importa quantos dados existam — um prior confiante
    demais pode dominar até uma amostra grande.
    """
    n = len(data)
    sample_mean = float(np.mean(data))

    prior_precision = 1.0 / prior_std**2
    data_precision = n / sigma**2

    return (prior_mean * prior_precision + sample_mean * data_precision) / (
        prior_precision + data_precision
    )
