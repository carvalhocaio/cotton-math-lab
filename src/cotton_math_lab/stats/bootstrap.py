"""Bootstrap: confidence intervals without assuming any closed form."""

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
    """Percentile bootstrap confidence interval.

    Resamples `data` WITH replacement `n_resamples` times, computes
    `statistic` on each resample, and returns the (α/2, 1-α/2)
    percentiles of the resulting distribution. Assumes no parametric
    form for `statistic` — it works for the mean, but also for the
    median, correlation, variance ratio, or any statistic without a
    closed-form standard error.
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
