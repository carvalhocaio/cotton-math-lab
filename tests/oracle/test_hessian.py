import numpy as np
import pytest
import torch

from cotton_math_lab.autodiff.gradient import gradient
from cotton_math_lab.autodiff.hessian import hessian


@pytest.mark.oracle
def test_gradient_matches_torch():
    def f(xs):
        x0, x1 = xs
        return x0**2 * x1 + x1**3

    x0 = np.array([1.2, -0.7])
    mine = gradient(f, x0)

    tx = torch.tensor(x0, requires_grad=True, dtype=torch.float64)
    (tx[0] ** 2 * tx[1] + tx[1] ** 3).backward()

    np.testing.assert_allclose(mine, tx.grad.numpy())


@pytest.mark.oracle
def test_hessian_matches_torch():
    def f(xs):
        x0, x1, x2 = xs
        return x0**2 * x1 + x1**3 - x0 * x2**2

    x0 = np.array([1.2, -0.7, 0.5])
    mine = hessian(f, x0)

    tx = torch.tensor(x0, requires_grad=True, dtype=torch.float64)

    def f_torch(x):
        return x[0] ** 2 * x[1] + x[1] ** 3 - x[0] * x[2] ** 2

    reference = torch.autograd.functional.hessian(f_torch, tx).numpy()
    np.testing.assert_allclose(mine, reference, atol=1e-6)


@pytest.mark.unit
def test_hessian_is_symmetric():
    """Teorema de Schwarz: derivadas parciais mistas comutam para f suave."""

    def f(xs):
        x0, x1, x2 = xs
        return x0**2 * x1 + x1**3 - x0 * x2**2

    h = hessian(f, np.array([1.2, -0.7, 0.5]))
    np.testing.assert_allclose(h, h.T, atol=1e-8)


@pytest.mark.unit
def test_hessian_of_quadratic_form_is_constant():
    """f(x) = ½xᵀAx tem Hessiana = A, constante em qualquer ponto — a
    definição mesma de curvatura constante de uma forma quadrática."""
    a = np.array([[4.0, 1.0], [1.0, 2.0]])

    def f(xs):
        x0, x1 = xs
        return (a[0, 0] * x0 * x0 + 2 * a[0, 1] * x0 * x1 + a[1, 1] * x1 * x1) * 0.5

    for point in (np.array([0.0, 0.0]), np.array([3.0, -2.0]), np.array([-1.5, 4.0])):
        np.testing.assert_allclose(hessian(f, point), a, atol=1e-4)
