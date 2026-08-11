"""Entropia, cross-entropy e divergência KL — estimadores discretos."""

import numpy as np


def entropy_discrete(p: np.ndarray, base: float | None = None) -> float:
    """H(p) = -Σ p(x)·log p(x). Convenção: 0·log(0) := 0 (o limite existe
    e vale 0, então categorias de probabilidade zero simplesmente não
    contribuem, em vez de gerar log(0) = -∞)."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    if base is not None:
        h /= np.log(base)
    return float(h)


def cross_entropy_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """H(p,q) = -Σ p(x)·log q(x) — o custo médio de codificar amostras de
    p usando um código otimizado para q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(q[mask])))


def kl_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p‖q) = Σ p(x)·log(p(x)/q(x)) = H(p,q) - H(p) — o custo EXTRA
    de usar o código de q em vez do código ótimo de p. Sempre ≥ 0
    (desigualdade de Gibbs), zero se e somente se p = q."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
