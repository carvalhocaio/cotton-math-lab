import pytest
import torch

from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.oracle
def test_power_gradient_matches_torch():
    x = Tensor(3.0)
    y = x**3
    y.backward()

    tx = torch.tensor(3.0, requires_grad=True)
    (tx**3).backward()

    assert x.grad == pytest.approx(tx.grad.item())


@pytest.mark.oracle
def test_exp_gradient_matches_torch():
    x = Tensor(1.5)
    y = x.exp()
    y.backward()

    tx = torch.tensor(1.5, requires_grad=True)
    torch.exp(tx).backward()

    assert x.grad == pytest.approx(tx.grad.item(), rel=1e-5)


@pytest.mark.oracle
def test_reverse_subtraction_gradient_matches_torch():
    """5 - x — exercita __rsub__, não só __sub__."""
    x = Tensor(2.0)
    y = 5.0 - x
    y.backward()

    tx = torch.tensor(2.0, requires_grad=True)
    (5.0 - tx).backward()

    assert x.grad == pytest.approx(tx.grad.item())


@pytest.mark.oracle
def test_division_gradient_matches_torch():
    """Divisão não tem regra própria — é mul + pow(-1) compostos."""
    p, q = Tensor(6.0), Tensor(3.0)
    r = p / q
    r.backward()

    tp = torch.tensor(6.0, requires_grad=True)
    tq = torch.tensor(3.0, requires_grad=True)
    (tp / tq).backward()

    assert p.grad == pytest.approx(tp.grad.item())
    assert q.grad == pytest.approx(tq.grad.item())


@pytest.mark.oracle
def test_nonlinear_composite_gradient_matches_torch():
    """f = exp(a*b) + a — encadeia produto, exp e soma num só grafo."""
    a, b = Tensor(0.5), Tensor(-1.2)
    f = (a * b).exp() + a
    f.backward()

    ta = torch.tensor(0.5, requires_grad=True)
    tb = torch.tensor(-1.2, requires_grad=True)
    (torch.exp(ta * tb) + ta).backward()

    assert a.grad == pytest.approx(ta.grad.item(), rel=1e-5)
    assert b.grad == pytest.approx(tb.grad.item(), rel=1e-5)


@pytest.mark.unit
def test_pow_rejects_non_scalar_exponent():
    from cotton_math_lab.exceptions import AutodiffError

    x = Tensor(2.0)
    with pytest.raises(AutodiffError, match="escalar"):
        x ** Tensor(2.0)


@pytest.mark.oracle
def test_log_gradient_matches_torch():
    x = Tensor(2.5)
    y = x.log()
    y.backward()

    tx = torch.tensor(2.5, requires_grad=True)
    torch.log(tx).backward()

    assert x.grad == pytest.approx(tx.grad.item())


@pytest.mark.oracle
def test_sigmoid_cross_entropy_composition_matches_torch():
    """Sigmoide não é primitiva — é 1/(1+exp(-z)), pura composição.
    Cross-entropy binária usa log em cascata com ela. Cobre os dois rótulos,
    porque cada um exercita um ramo diferente da soma (y·log(p) vs (1-y)·log(1-p))."""

    def sigmoid(t):
        return 1.0 / (1.0 + (-t).exp())

    def bce_loss(weight, x_input, bias, y_true):
        z = weight * x_input + bias
        p = sigmoid(z)
        return -(y_true * p.log() + (1 - y_true) * (1.0 - p).log())

    for y_true in (1.0, 0.0):
        w, b = Tensor(0.8), Tensor(-0.2)
        loss = bce_loss(w, Tensor(1.3), b, y_true)
        loss.backward()

        tw = torch.tensor(0.8, requires_grad=True)
        tb = torch.tensor(-0.2, requires_grad=True)
        tz = tw * 1.3 + tb
        tp = torch.sigmoid(tz)
        tloss = -(y_true * torch.log(tp) + (1 - y_true) * torch.log(1 - tp))
        tloss.backward()

        assert loss.data == pytest.approx(tloss.item())
        assert w.grad == pytest.approx(tw.grad.item())
        assert b.grad == pytest.approx(tb.grad.item())
