"""Verificação de gradientes por diferenças finitas centrais.

Serve como oráculo independente de qualquer motor de autodiff — não depende
do torch existir nem de nenhuma outra biblioteca, só da definição de
derivada. É o "oráculo dos oráculos": qualquer regra de `_backward` nova
implementada no motor pode ser validada contra este utilitário sozinho.
"""

import numpy as np


def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Gradiente de f em x via diferença finita central, componente a componente.

    (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h) aproxima ∂f/∂xᵢ com erro de
    truncamento O(h²). Mas h pequeno demais introduz cancelamento
    catastrófico: f(x+h) e f(x-h) ficam quase iguais, e a subtração perde
    dígitos significativos. O ponto ótimo empírico para double precision
    fica perto de h ≈ 1e-5, onde os dois erros — truncamento e
    cancelamento — se equilibram.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)

    iterator = np.nditer(x, flags=["multi_index"])
    for _ in iterator:
        idx = iterator.multi_index
        original = x[idx]

        x[idx] = original + h
        f_plus = f(x)

        x[idx] = original - h
        f_minus = f(x)

        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2.0 * h)

    return grad
