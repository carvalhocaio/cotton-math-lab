import numpy as np
import pytest
import torch

from cotton_math_lab.autodiff.optim import Adam
from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.oracle
def test_adam_trajectory_matches_torch():
    x, y = Tensor(3.0), Tensor(3.0)
    optimizer = Adam([x, y], lr=0.1)

    tx = torch.tensor(3.0, requires_grad=True)
    ty = torch.tensor(3.0, requires_grad=True)
    torch_optimizer = torch.optim.Adam([tx, ty], lr=0.1)

    for _ in range(15):
        optimizer.zero_grad()
        loss = x**2 + y**2 * 10.0
        loss.backward()
        optimizer.step()

        torch_optimizer.zero_grad()
        torch_loss = tx**2 + ty**2 * 10.0
        torch_loss.backward()
        torch_optimizer.step()

    assert x.data == pytest.approx(tx.item(), abs=1e-5)
    assert y.data == pytest.approx(ty.item(), abs=1e-5)


@pytest.mark.unit
def test_bias_correction_makes_first_step_exactly_lr_sized():
    """Em t=1, m̂ = grad e v̂ = grad² exatamente (a correção cancela o
    fator (1-β) que entrou em m e v) — logo m̂/√v̂ = sign(grad), e o
    primeiro passo do Adam é sempre ±lr, INDEPENDENTE da magnitude do
    gradiente. É um invariante matemático, não uma aproximação."""
    lr = 0.1

    for gradient_value in (0.001, 6.0, 500.0, -3.0):
        param = Tensor(0.0)
        param.grad = np.array(gradient_value)
        optimizer = Adam([param], lr=lr)
        optimizer.step()

        step_taken = -float(param.data)  # deslocamento em relação a 0.0
        assert abs(step_taken) == pytest.approx(lr, rel=1e-4)
        assert np.sign(step_taken) == np.sign(gradient_value)


@pytest.mark.unit
def test_uncorrected_first_step_is_larger_by_known_factor():
    """O contraponto: SEM correção de viés, o primeiro passo não é ±lr —
    fica inflado por (1-β₁)/√(1-β₂), um fator fixo e previsível a partir
    dos betas, não do gradiente. A raiz só se aplica ao lado de v (que já
    carrega grad²); o lado de m entra sem raiz nenhuma — daí a assimetria."""
    beta1, beta2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    grad = 6.0

    m = (1 - beta1) * grad
    v = (1 - beta2) * grad**2
    uncorrected_step = lr * m / (np.sqrt(v) + eps)

    expected_inflation = (1 - beta1) / np.sqrt(1 - beta2)
    assert abs(uncorrected_step) == pytest.approx(lr * expected_inflation, rel=1e-3)
