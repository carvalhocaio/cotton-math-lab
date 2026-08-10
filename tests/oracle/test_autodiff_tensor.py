import numpy as np
import pytest
import torch

from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.oracle
def test_addition_gradient_matches_torch():
    a, b = Tensor(2.0), Tensor(3.0)
    out = a + b
    out.backward()

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    (ta + tb).backward()

    assert a.grad == pytest.approx(ta.grad.item())
    assert b.grad == pytest.approx(tb.grad.item())


@pytest.mark.oracle
def test_multiplication_gradient_matches_torch():
    a, b = Tensor(2.0), Tensor(-3.0)
    out = a * b
    out.backward()

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(-3.0, requires_grad=True)
    (ta * tb).backward()

    assert a.grad == pytest.approx(ta.grad.item())
    assert b.grad == pytest.approx(tb.grad.item())


@pytest.mark.oracle
def test_composite_expression_matches_torch():
    """f = a*b + b*c + a*c — cobre soma, produto e múltiplos caminhos no grafo."""
    values = (2.0, -3.0, 4.0)
    a, b, c = (Tensor(v) for v in values)
    f = a * b + b * c + a * c
    f.backward()

    ta, tb, tc = (torch.tensor(v, requires_grad=True) for v in values)
    tf = ta * tb + tb * tc + ta * tc
    tf.backward()

    assert a.grad == pytest.approx(ta.grad.item())
    assert b.grad == pytest.approx(tb.grad.item())
    assert c.grad == pytest.approx(tc.grad.item())


@pytest.mark.oracle
def test_reused_variable_accumulates_gradient():
    """y = x*x + x — x aparece em dois caminhos.

    O gradiente deve SOMAR, não sobrescrever.
    """
    x = Tensor(3.0)
    y = x * x + x
    y.backward()

    tx = torch.tensor(3.0, requires_grad=True)
    ty = tx * tx + tx
    ty.backward()

    assert x.grad == pytest.approx(tx.grad.item())


@pytest.mark.oracle
def test_elementwise_array_gradient_matches_torch():
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(4)
    b_np = rng.standard_normal(4)

    a, b = Tensor(a_np), Tensor(b_np)
    out = a * b + b
    out.backward()  # seed = ones_like(out), equivalente a backward de out.sum()

    ta = torch.tensor(a_np, requires_grad=True)
    tb = torch.tensor(b_np, requires_grad=True)
    (ta * tb + tb).sum().backward()

    np.testing.assert_allclose(a.grad, ta.grad.numpy())
    np.testing.assert_allclose(b.grad, tb.grad.numpy())


@pytest.mark.oracle
def test_sum_gradient_matches_torch():
    rng = np.random.default_rng(0)
    w_np = rng.standard_normal(8)
    x_np = rng.standard_normal(8)

    w = Tensor(w_np.copy())
    z = (w * Tensor(x_np)).sum() + Tensor(0.5)
    z.backward()

    tw = torch.tensor(w_np, requires_grad=True)
    tz = (tw * torch.tensor(x_np)).sum() + 0.5
    tz.backward()

    assert z.data == pytest.approx(tz.item())
    np.testing.assert_allclose(w.grad, tw.grad.numpy())


@pytest.mark.unit
def test_root_gradient_is_one():
    """dy/dy = 1, sempre — é o valor de partida que a regra da cadeia propaga."""
    x = Tensor(5.0)
    x.backward()
    assert x.grad == pytest.approx(1.0)
