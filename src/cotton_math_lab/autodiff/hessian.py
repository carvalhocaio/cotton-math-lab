"""Hessian via central finite difference of the analytic gradient."""

import numpy as np

from cotton_math_lab.autodiff.gradient import gradient


def hessian(f, x0: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Hessian of f: R^n → R at x0.

    H[:, j] ≈ (∇f(x0 + h·eⱼ) - ∇f(x0 - h·eⱼ)) / (2h) — central finite
    difference applied to the GRADIENT, which is already exact (it came
    from autodiff, not from another finite difference). This is
    equivalent to the "Jacobian of the gradient" computed via finite
    differences: one derivative is exact (via `Tensor`), the other is
    numerical (via this loop) — a deliberate hybrid, cheaper than a full
    second-order autodiff.
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
