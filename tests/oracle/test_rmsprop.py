import pytest
import torch

from cotton_math_lab.autodiff.optim import SGD, RMSProp
from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.oracle
def test_rmsprop_trajectory_matches_torch():
    x, y = Tensor(3.0), Tensor(3.0)
    optimizer = RMSProp([x, y], lr=0.1)

    tx = torch.tensor(3.0, requires_grad=True)
    ty = torch.tensor(3.0, requires_grad=True)
    torch_optimizer = torch.optim.RMSprop([tx, ty], lr=0.1)

    for _ in range(15):
        optimizer.zero_grad()
        loss = x**2 + y**2 * 10.0
        loss.backward()
        optimizer.step()

        torch_optimizer.zero_grad()
        torch_loss = tx**2 + ty**2 * 10.0
        torch_loss.backward()
        torch_optimizer.step()

    assert x.data == pytest.approx(tx.item(), abs=1e-6)
    assert y.data == pytest.approx(ty.item(), abs=1e-6)


@pytest.mark.unit
def test_rmsprop_equalizes_step_size_across_different_curvatures():
    """O ponto central do método: apesar de y ter curvatura 10× maior que
    x (logo, gradiente inicial 10× maior), RMSProp normaliza o passo
    efetivo — os dois avançam quase igual. SGD não tem essa propriedade:
    a direção de maior curvatura sempre anda muito mais."""

    def displacement_ratio(optimizer_cls, steps=3, **kwargs):
        x, y = Tensor(3.0), Tensor(3.0)
        optimizer = optimizer_cls([x, y], **kwargs)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = x**2 + y**2 * 10.0
            loss.backward()
            optimizer.step()
        dx = 3.0 - float(x.data)
        dy = 3.0 - float(y.data)
        return dy / dx

    sgd_ratio = displacement_ratio(SGD, lr=0.02)
    rmsprop_ratio = displacement_ratio(RMSProp, lr=0.1)

    assert sgd_ratio > 5.0  # SGD: y anda MUITO mais que x
    assert rmsprop_ratio == pytest.approx(1.0, abs=0.05)  # RMSProp: quase igual


@pytest.mark.unit
def test_rmsprop_converges_across_a_wide_lr_range():
    """Ao contrário de Momentum (janela estreita de lr estável), RMSProp
    converge bem numa faixa ampla — a adaptação por parâmetro compensa
    boa parte da escolha de lr."""

    def final_loss(lr, steps=40):
        x, y = Tensor(3.0), Tensor(3.0)
        optimizer = RMSProp([x, y], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = x**2 + y**2 * 10.0
            loss.backward()
            optimizer.step()
        return float(x.data**2 + y.data**2 * 10.0)

    for lr in (0.1, 0.2, 0.3):
        assert final_loss(lr) < 1e-4
