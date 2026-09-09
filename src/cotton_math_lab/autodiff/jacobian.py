"""Jacobian via multiple reverse-mode passes, one per output."""

import numpy as np

from cotton_math_lab.autodiff.dual import Dual
from cotton_math_lab.autodiff.tensor import Tensor


def jacobian(f, x0: np.ndarray) -> np.ndarray:
    """Jacobian via m reverse passes, one per output."""
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
    """Jacobian via n forward passes, one per input direction.

    Each pass seeds the dual part of ONE input with 1.0 (the rest with
    0.0) and, at the end, reads the dual part of ALL outputs at once —
    the exact mirror of reverse mode: there, one pass gives an entire
    row (all inputs, one output); here, one pass gives an entire column
    (all outputs, one input).
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
