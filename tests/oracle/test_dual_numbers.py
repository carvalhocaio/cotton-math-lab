import numpy as np
import pytest
import torch

from cotton_math_lab.autodiff.dual import Dual
from cotton_math_lab.autodiff.jacobian import jacobian, jacobian_forward


@pytest.mark.unit
def test_dual_multiplication_follows_product_rule():
    """d/dx(x*x) at x=3 is 2x=6 - the dual part carries the derivative."""
    x = Dual(3.0, dual=1.0)
    result = x * x
    assert result.real == pytest.approx(9.0)
    assert result.dual == pytest.approx(6.0)


@pytest.mark.oracle
def test_jacobian_forward_matches_torch_square_case():
    def f(xs):
        x0, x1, x2 = xs
        return [x0 * x1, x1**2 + x2.exp()]

    x0 = np.array([0.6, -1.2, 0.3])
    mine = jacobian_forward(f, x0)

    tx = torch.tensor(x0, requires_grad=True, dtype=torch.float64)

    def f_torch(x):
        return torch.stack([x[0] * x[1], x[1] ** 2 + torch.exp(x[2])])

    reference = torch.autograd.functional.jacobian(f_torch, tx).numpy()
    np.testing.assert_allclose(mine, reference, atol=1e-10)


@pytest.mark.unit
def test_forward_and_reverse_modes_agree():
    """Two independent implementations of the same Jacobian - if they
    agree, that's strong evidence both are correct."""

    def f(xs):
        x0, x1 = xs
        return [x0 + x1, x0 * x1, x0**2]

    x0 = np.array([1.3, -0.4])
    np.testing.assert_allclose(jacobian_forward(f, x0), jacobian(f, x0), atol=1e-10)


@pytest.mark.unit
def test_reverse_mode_wins_when_outputs_are_few():
    """n=6 inputs, m=1 output - reverse should win by a wide margin."""
    calls = {"forward": 0, "reverse": 0}

    def f_wide(xs):
        total = xs[0]
        for x in xs[1:]:
            total = total + x * x
        return [total]

    def counted_forward(xs):
        calls["forward"] += 1
        return f_wide(xs)

    def counted_reverse(xs):
        calls["reverse"] += 1
        return f_wide(xs)

    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    jacobian_forward(counted_forward, x0)
    jacobian(counted_reverse, x0)

    assert calls["forward"] == 6
    assert calls["reverse"] == 1


@pytest.mark.unit
def test_forward_mode_wins_when_inputs_are_few():
    """n=2 inputs, m=5 outputs - the advantage flips."""
    calls = {"forward": 0, "reverse": 0}

    def f_tall(xs):
        x0, x1 = xs
        return [x0 + x1, x0 - x1, x0 * x1, x0**2, x1**2]

    def counted_forward(xs):
        calls["forward"] += 1
        return f_tall(xs)

    def counted_reverse(xs):
        calls["reverse"] += 1
        return f_tall(xs)

    x0 = np.array([1.5, -0.5])
    jacobian_forward(counted_forward, x0)
    jacobian(counted_reverse, x0)

    assert calls["forward"] == 2
    assert calls["reverse"] == 5
