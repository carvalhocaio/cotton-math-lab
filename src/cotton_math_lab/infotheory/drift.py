"""JS as a drift detector — symmetric and bounded, unlike KL."""

import numpy as np

from cotton_math_lab.infotheory.discrete import kl_divergence_discrete


def js_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """D_JS(p,q) = ½·D_KL(p‖m) + ½·D_KL(q‖m), with m = (p+q)/2.

    Unlike KL, it's symmetric and bounded (0 ≤ D_JS ≤ log 2) —
    properties that make it a better choice than KL for measuring "how
    different" two distributions are, even though it isn't a formal
    metric itself (the square root, not D_JS itself, satisfies the
    triangle inequality — that's the Jensen-Shannon distance).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_discrete(p, m) + 0.5 * kl_divergence_discrete(q, m)


def histogram_distribution(
    values: np.ndarray, bin_edges: np.ndarray, smoothing: float = 1e-6
) -> np.ndarray:
    """Continuous sample -> discrete distribution via histogram, with
    minimal smoothing to avoid absolute zero probability (which would
    break KL/JS via log(0))."""
    counts, _ = np.histogram(values, bins=bin_edges)
    counts = counts.astype(np.float64) + smoothing
    return counts / counts.sum()


def detect_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 20,
    threshold: float = 0.05,
) -> tuple[bool, float]:
    """Compares the distribution of `current` against `baseline` via JS.

    Returns (drift_detected, js_divergence). The threshold is a design
    choice, not a universal value — calibratable from historical data on
    how much natural variation between genuinely similar seasons
    typically produces.
    """
    combined = np.concatenate([baseline, current])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    p = histogram_distribution(baseline, bin_edges)
    q = histogram_distribution(current, bin_edges)

    divergence = js_divergence_discrete(p, q)
    return divergence > threshold, divergence
