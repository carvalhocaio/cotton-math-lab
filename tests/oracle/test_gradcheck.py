import numpy as np
import pytest

from cotton_math_lab.autodiff.gradcheck import numerical_gradient
from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.oracle
def test_polynomial_gradient_matches_finite_differences():
    x0 = np.array([1.5, -2.0, 0.7])

    x = Tensor(x0.copy())
    out = x**3 + x * 2.0 - 1.0
    out.backward()

    numeric = numerical_gradient(lambda v: np.sum(v**3 + v * 2.0 - 1.0), x0.copy())
    np.testing.assert_allclose(x.grad, numeric, rtol=1e-6, atol=1e-8)


@pytest.mark.oracle
def test_exp_gradient_matches_finite_differences():
    x0 = np.array([0.2, -1.3, 2.5])

    x = Tensor(x0.copy())
    out = x.exp() * x
    out.backward()

    numeric = numerical_gradient(lambda v: np.sum(np.exp(v) * v), x0.copy())
    np.testing.assert_allclose(x.grad, numeric, rtol=1e-6, atol=1e-8)


@pytest.mark.oracle
def test_multivariate_gradient_matches_finite_differences():
    """f(a,b) = sum(a*b + exp(a)) — cada entrada tem sua própria derivada parcial."""
    a0 = np.array([0.3, -0.5, 1.1])
    b0 = np.array([2.0, 1.0, -0.7])

    a, b = Tensor(a0.copy()), Tensor(b0.copy())
    out = a * b + a.exp()
    out.backward()

    numeric_a = numerical_gradient(lambda v: np.sum(v * b0 + np.exp(v)), a0.copy())
    numeric_b = numerical_gradient(lambda v: np.sum(a0 * v + np.exp(a0)), b0.copy())

    np.testing.assert_allclose(a.grad, numeric_a, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(b.grad, numeric_b, rtol=1e-6, atol=1e-8)


@pytest.mark.unit
def test_h_too_small_degrades_accuracy_via_cancellation():
    """
    A curva em U: h menor não é sempre melhor — cancelamento domina abaixo de ~1e-6.
    """
    x0 = np.array([1.5, -2.0, 0.7])
    exact = 3 * x0**2  # d/dx(x^3) = 3x^2, conhecido em forma fechada

    good = numerical_gradient(lambda v: np.sum(v**3), x0.copy(), h=1e-5)
    too_small = numerical_gradient(lambda v: np.sum(v**3), x0.copy(), h=1e-12)

    error_good = np.abs(good - exact).max()
    error_too_small = np.abs(too_small - exact).max()

    assert error_good < error_too_small
