"""Gradiente analítico de f: R^n -> R via um único backward()."""

import numpy as np

from cotton_math_lab.autodiff.tensor import Tensor


def gradient(f, x0: np.ndarray) -> np.ndarray:
    """Gradiente  exato de f em x0.

    `f` recebe uma lista de n Tensores escalares e devolve UM Tensor
    escalar. Como a saída é escalar (m=1), uma úniica passada reversa basta
    - é o caso degenerado de `jacobian` quando m=1, mas vale ter a função
    própria: fica explícito que a saída é vetor, não matriz linha, e é o
    bloco de construção do módulo de Hessian logo abaixo.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    inputs = [Tensor(value) for value in x0]
    output = f(inputs)
    output.backward()
    return np.array([tensor.grad for tensor in inputs])
