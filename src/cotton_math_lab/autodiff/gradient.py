"""Analytic gradient of f: R^n -> R via a single backward()."""

import numpy as np

from cotton_math_lab.autodiff.tensor import Tensor


def gradient(f, x0: np.ndarray) -> np.ndarray:
    """Exact gradient of f at x0.

    `f` receives a list of n scalar Tensors and returns ONE scalar
    Tensor. Since the output is scalar (m=1), a single reverse pass is
    enough — it's the degenerate case of `jacobian` when m=1, but it's
    worth having its own function: it makes explicit that the output is
    a vector, not a row matrix, and it's the building block for the
    Hessian module right below.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    inputs = [Tensor(value) for value in x0]
    output = f(inputs)
    output.backward()
    return np.array([tensor.grad for tensor in inputs])
