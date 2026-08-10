import numpy as np
import pytest
import torch

from cotton_math_lab.autodiff.jacobian import jacobian
from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.oracle
def test_jacobian_matches_torch_square_case():
    """f: R^3 -> R^2 — [x0*x1, x1^2 + exp(x2)]."""

    def f(xs):
        x0, x1, x2 = xs
        return [x0 * x1, x1**2 + x2.exp()]

    x0 = np.array([0.6, -1.2, 0.3])
    mine = jacobian(f, x0)

    tx = torch.tensor(x0, requires_grad=True, dtype=torch.float64)

    def f_torch(x):
        return torch.stack([x[0] * x[1], x[1] ** 2 + torch.exp(x[2])])

    reference = torch.autograd.functional.jacobian(f_torch, tx).numpy()
    np.testing.assert_allclose(mine, reference, atol=1e-10)


@pytest.mark.oracle
def test_jacobian_matches_torch_rectangular_case():
    """f: R^2 -> R^3, mais saídas que entradas."""

    def g(xs):
        x0, x1 = xs
        return [x0 + x1, x0 * x1, x0**2]

    x0 = np.array([1.3, -0.4])
    mine = jacobian(g, x0)

    tx = torch.tensor(x0, requires_grad=True, dtype=torch.float64)

    def g_torch(x):
        return torch.stack([x[0] + x[1], x[0] * x[1], x[0] ** 2])

    reference = torch.autograd.functional.jacobian(g_torch, tx).numpy()
    np.testing.assert_allclose(mine, reference, atol=1e-10)


@pytest.mark.unit
def test_jacobian_row_matches_independent_backward_call():
    """Propriedade que define o Jacobiano: linha i = gradiente de saída_i sozinha."""

    def f(xs):
        x0, x1 = xs
        return [x0 * x1, x0**2 - x1]

    x0 = np.array([2.0, 3.0])
    full_jacobian = jacobian(f, x0)

    a, b = Tensor(x0[0]), Tensor(x0[1])
    _, second_output = f([a, b])
    second_output.backward()

    np.testing.assert_allclose(full_jacobian[1], [a.grad, b.grad])


@pytest.mark.unit
def test_zero_grad_resets_to_zero():
    x = Tensor(5.0)
    x.grad = np.array(99.0)
    x.zero_grad()
    assert x.grad == pytest.approx(0.0)
