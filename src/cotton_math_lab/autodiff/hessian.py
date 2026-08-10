"""Hessiana via diferença finita central do gradiente analítico."""

import numpy as np

from cotton_math_lab.autodiff.gradient import gradient


def hessian(f, x0: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Hessiana de f: R^n → R em x0.

    H[:, j] ≈ (∇f(x0 + h·eⱼ) - ∇f(x0 - h·eⱼ)) / (2h) — diferença finita
    central aplicada ao GRADIENTE, que já é exato (veio de autodiff, não
    de outra diferença finita). Isso equivale a "Jacobiano do gradiente"
    calculado por diferenças finitas: uma derivada é exata (via `Tensor`),
    a outra é numérica (via este laço) — um híbrido deliberado, mais barato
    que autodiff de segunda ordem completo.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n = len(x0)
    result = np.zeros((n, n))

    for col in range(n):
        forward = x0.copy()
        forward[col] += h
        backward = x0.copy()
        backward[col] -= h

        result[:, col] = (gradient(f, forward) - gradient(f, backward)) / (2.0 * h)

    return result
