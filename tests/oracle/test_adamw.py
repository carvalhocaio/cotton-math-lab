import numpy as np
import pytest
import torch

from cotton_math_lab.autodiff.optim import Adam, AdamW
from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.oracle
def test_adamw_trajectory_matches_torch():
    x, y = Tensor(3.0), Tensor(3.0)
    optimizer = AdamW([x, y], lr=0.1, weight_decay=0.1)

    tx = torch.tensor(3.0, requires_grad=True)
    ty = torch.tensor(3.0, requires_grad=True)
    torch_optimizer = torch.optim.AdamW([tx, ty], lr=0.1, weight_decay=0.1)

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


def _build_history_then_observe_pure_decay(optimizer_cls, weight_decay, **kwargs):
    """Duas fases: primeiro constrói históricos de v MUITO diferentes pra
    dois parâmetros (um recebe gradiente 10.000× maior que o outro); depois
    zera o gradiente real e observa só o efeito do weight decay."""
    a, b = Tensor(2.0), Tensor(2.0)
    optimizer = optimizer_cls([a, b], lr=0.05, weight_decay=weight_decay, **kwargs)

    for _ in range(20):
        optimizer.zero_grad()
        a.grad = np.array(10.0)
        b.grad = np.array(0.001)
        optimizer.step()
    a_before, b_before = float(a.data), float(b.data)

    for _ in range(20):
        optimizer.zero_grad()
        a.grad = np.array(0.0)
        b.grad = np.array(0.0)
        optimizer.step()

    decay_a = (a_before - float(a.data)) / a_before
    decay_b = (b_before - float(b.data)) / b_before
    return decay_a, decay_b


@pytest.mark.unit
def test_adamw_decay_is_uniform_regardless_of_gradient_history():
    """O ponto central do ciclo: com AdamW, dois parâmetros com histórico
    de gradiente radicalmente diferente decaem na MESMA proporção — o
    weight decay não é distorcido pela escala adaptativa do Adam."""
    decay_a, decay_b = _build_history_then_observe_pure_decay(AdamW, weight_decay=0.1)
    assert abs(decay_a - decay_b) < 0.01


@pytest.mark.unit
def test_naive_l2_in_gradient_makes_decay_depend_on_history():
    """O contraponto que prova por que AdamW existe: somando L2 ao
    gradiente ANTES do Adam processar (a forma ingênua de 'Adam + weight
    decay'), a mesma força nominal produz decaimento bem diferente entre
    os dois parâmetros — o histórico de v distorce a regularização."""

    class AdamWithL2InGradient:
        def __init__(
            self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        ):
            self.parameters = parameters
            self.lr, self.eps, self.weight_decay = lr, eps, weight_decay
            self.beta1, self.beta2 = betas
            self.m = [np.zeros_like(p.data) for p in parameters]
            self.v = [np.zeros_like(p.data) for p in parameters]
            self.t = 0

        def step(self):
            self.t += 1
            for p, m, v in zip(self.parameters, self.m, self.v, strict=True):
                effective_grad = p.grad + self.weight_decay * p.data
                m[...] = self.beta1 * m + (1 - self.beta1) * effective_grad
                v[...] = self.beta2 * v + (1 - self.beta2) * (effective_grad**2)
                m_hat = m / (1 - self.beta1**self.t)
                v_hat = v / (1 - self.beta2**self.t)
                p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        def zero_grad(self):
            for p in self.parameters:
                p.zero_grad()

    decay_a, decay_b = _build_history_then_observe_pure_decay(
        AdamWithL2InGradient, weight_decay=0.1
    )
    assert abs(decay_a - decay_b) > 0.1
