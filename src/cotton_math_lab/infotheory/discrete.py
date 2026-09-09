"""Entropy, cross-entropy, and KL divergence — discrete estimators."""

import numpy as np


def entropy_discrete(p: np.ndarray, base: float | None = None) -> float:
    """H(p) = -Σ p(x)·log p(x). Convention: 0·log(0) := 0 (the limit
    exists and equals 0, so zero-probability categories simply don't
    contribute, instead of producing log(0) = -∞)."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    if base is not None:
        h /= np.log(base)
    return float(h)


def cross_entropy_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """H(p,q) = -Σ p(x)·log q(x) — the average cost of encoding samples
    from p using a code optimized for q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(q[mask])))


def kl_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p‖q) = Σ p(x)·log(p(x)/q(x)) = H(p,q) - H(p) — the EXTRA
    cost of using q's code instead of p's optimal code. Always ≥ 0
    (Gibbs' inequality), zero if and only if p = q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
