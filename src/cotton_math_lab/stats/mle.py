"""Estimação de máxima verossimilhança (MLE)."""

import numpy as np


def mle_normal(data: np.ndarray) -> tuple[float, float]:
    """MLE de μ e σ de uma amostra ~ Normal(μ, σ²).

    Maximizar a log-verossimilhança de uma Normal tem solução fechada:
    μ_MLE é a média amostral — sempre não-enviesado, para qualquer n.
    σ²_MLE é a variância amostral dividida por n, NÃO por n-1 (a "correção
    de Bessel" usada em quase todo outro contexto estatístico). Essa
    escolha não é capricho: sai diretamente de zerar a derivada da
    log-verossimilhança em relação a σ², sem nenhum ajuste post-hoc de
    viés — e por isso σ²_MLE subestima a variância populacional na razão
    exata (n-1)/n.
    """
    mu_hat = float(np.mean(data))
    sigma_hat = float(np.sqrt(np.mean((data - mu_hat) ** 2)))
    return mu_hat, sigma_hat
