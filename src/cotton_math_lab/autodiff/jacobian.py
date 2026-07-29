"""Jacobiano via múltiplas passadas de modo reverso, uma por saída."""

import numpy as np

from cotton_math_lab.autodiff.dual import Dual
from cotton_math_lab.autodiff.tensor import Tensor


def jacobian(f, x0: np.ndarray) -> np.ndarray:
    """Jacobiano via m passadas reversas, uma por saída."""
    x0 = np.asarray(x0, dtype=np.float64)

    n_outputs = None
    rows = []
    row = 0
    while n_outputs is None or row < n_outputs:
        inputs = [Tensor(value) for value in x0]
        outputs = f(inputs)
        if n_outputs is None:
            n_outputs = len(outputs)
        outputs[row].backward()
        rows.append([tensor.grad for tensor in inputs])
        row += 1

    return np.array(rows)


def jacobian_forward(f, x0: np.ndarray) -> np.ndarray:
    """Jacobiano via n passadas forward, uma por direção de entrada.

    Cada passada semeia a parte dual de UMA entrada com 1.0 (as demais em
    0.0) e lê, no fim, a parte dual de TODAS as saídas de uma vez — o
    espelho exato do modo reverso: lá, uma passada dá uma linha inteira
    (todas as entradas, uma saída); aqui, uma passada dá uma coluna
    inteira (todas as saídas, uma entrada).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n_inputs = len(x0)

    result = None
    for col in range(n_inputs):
        duals = [Dual(x0[i], dual=1.0 if i == col else 0.0) for i in range(n_inputs)]
        outputs = f(duals)
        if result is None:
            result = np.zeros((len(outputs), n_inputs))
        for row, output in enumerate(outputs):
            result[row, col] = output.dual

    return result
