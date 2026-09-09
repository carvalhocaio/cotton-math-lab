"""Gradient checking via central finite differences.

Serves as an oracle independent of any autodiff engine — it doesn't
depend on torch existing or on any other library, only on the
definition of the derivative. It's the "oracle of oracles": any new
`_backward` rule implemented in the engine can be validated against
this utility alone.
"""

import numpy as np


def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Gradient of f at x via central finite difference, component by component.

    (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h) approximates ∂f/∂xᵢ with
    truncation error O(h²). But h too small introduces catastrophic
    cancellation: f(x+h) and f(x-h) become nearly equal, and the
    subtraction loses significant digits. The empirical optimum for
    double precision is near h ≈ 1e-5, where the two errors —
    truncation and cancellation — balance out.
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
