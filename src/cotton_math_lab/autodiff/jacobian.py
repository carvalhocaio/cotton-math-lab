"""Jacobiano via múltiplas passadas de modo reverso, uma por saída."""

import numpy as np

from cotton_math_lab.autodiff.tensor import Tensor


def jacobian(f, x0: np.ndarray) -> np.ndarray:
    """Jacobiano de f: R^n → R^m via m passadas reversas.

    `f` recebe uma lista de n Tensores escalares e devolve uma lista de m
    Tensores escalares. Como este motor não retém o grafo entre chamadas de
    backward (não há `retain_graph` como no PyTorch), cada linha do
    Jacobiano reconstrói o grafo do zero a partir de `x0` — o preço de
    manter o motor simples é recomputar o forward pass m vezes.
    """
    x0 = np.asarray(x0, dtype=np.float64)

    probe_inputs = [Tensor(value) for value in x0]
    probe_outputs = f(probe_inputs)
    n_outputs, n_inputs = len(probe_outputs), len(x0)

    result = np.zeros((n_outputs, n_inputs))
    for row in range(n_outputs):
        inputs = [Tensor(value) for value in x0]
        outputs = f(inputs)
        outputs[row].backward()
        for col, tensor in enumerate(inputs):
            result[row, col] = tensor.grad

    return result
