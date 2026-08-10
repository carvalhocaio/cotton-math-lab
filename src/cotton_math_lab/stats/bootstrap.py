"""Bootstrap: intervalos de confiança sem assumir forma fechada nenhuma."""

from collections.abc import Callable

import numpy as np


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int,
) -> tuple[float, float]:
    """Intervalo de confiança por bootstrap percentil.

    Reamostra `data` COM reposição `n_resamples` vezes, calcula `statistic`
    em cada reamostra, e devolve os percentis (α/2, 1-α/2) da distribuição
    resultante. Não assume nenhuma forma paramétrica para `statistic` —
    funciona para a média, mas também para mediana, correlação, razão de
    variâncias, ou qualquer estatística sem fórmula fechada de erro-padrão.
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resamples = data[resample_indices]
    bootstrap_statistics = np.array([statistic(row) for row in resamples])

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(bootstrap_statistics, alpha))
    upper = float(np.quantile(bootstrap_statistics, 1.0 - alpha))
    return lower, upper
