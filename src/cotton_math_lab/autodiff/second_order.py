"""Newton and quasi-Newton (BFGS) — reuse gradient() and hessian() from
Module 2 directly. No new math, just a new application of the same pieces."""

import numpy as np

from cotton_math_lab.autodiff.gradient import gradient
from cotton_math_lab.autodiff.hessian import hessian as hessian_fn
from cotton_math_lab.autodiff.tensor import Tensor


def newton_minimize(
    f, x0: np.ndarray, max_iter: int = 50, tol: float = 1e-10
) -> tuple[np.ndarray, int]:
    """Newton's method: at each step, solves H·Δx = -g and steps by Δx.

    For a quadratic, the Hessian is constant and the step is EXACT — a
    single iteration finds the minimum, because the second-order
    quadratic approximation the method uses isn't an approximation in
    that case, it's the exact function. Outside of quadratics, each step
    minimizes the local quadratic approximation, and convergence tends
    to be extremely fast near the minimum — at the cost of assembling
    and solving an n×n linear system at every iteration.
    """
    x = np.asarray(x0, dtype=np.float64).copy()

    for iteration in range(1, max_iter + 1):
        g = gradient(f, x)
        if np.linalg.norm(g) < tol:
            return x, iteration - 1
        hessian = hessian_fn(f, x)
        x = x - np.linalg.solve(hessian, g)

    return x, max_iter


def bfgs_minimize(
    f, x0: np.ndarray, max_iter: int = 200, tol: float = 1e-8
) -> tuple[np.ndarray, int]:
    """Quasi-Newton BFGS: approximates the INVERSE of the Hessian using
    only gradients, without ever assembling the true Hessian.

    At each step, updates the H⁻¹ approximation using the secant
    equation — the observed change in the gradient informs curvature,
    without needing an explicit second derivative. Costs O(n²) per step,
    much less than the O(n) gradients plus O(n³) of solving a system
    that pure Newton requires every iteration — at the price of needing
    more iterations to converge with the same quality.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    n = len(x)
    h_inv = np.eye(n)
    g = gradient(f, x)
    iteration = 0

    while iteration < max_iter:
        if np.linalg.norm(g) < tol:
            break
        iteration += 1

        direction = -h_inv @ g

        step = 1.0
        f_x = float(f([Tensor(v) for v in x]).data)
        while step > 1e-12:
            x_new = x + step * direction
            f_new = float(f([Tensor(v) for v in x_new]).data)
            if f_new < f_x:
                break
            step *= 0.5
        x_new = x + step * direction
        g_new = gradient(f, x_new)

        s = x_new - x
        y = g_new - g
        sy = s @ y
        if (
            sy > 1e-10
        ):  # only update with positive curvature (keeps H⁻¹ positive-definite)
            rho = 1.0 / sy
            identity = np.eye(n)
            h_inv = (identity - rho * np.outer(s, y)) @ h_inv @ (
                identity - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)

        x, g = x_new, g_new

    return x, iteration
