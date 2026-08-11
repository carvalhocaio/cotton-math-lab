import numpy as np
import pytest
import torch

from cotton_math_lab.autodiff.benchmarks import rosenbrock
from cotton_math_lab.autodiff.optim import (
    SGD,
    Adam,
    Momentum,
    NesterovMomentum,
    RMSProp,
)
from cotton_math_lab.autodiff.tensor import Tensor


def _run(optimizer_cls, steps=2000, start=(-1.5, 2.0), **kwargs):
    x, y = Tensor(start[0]), Tensor(start[1])
    optimizer = optimizer_cls([x, y], **kwargs)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = rosenbrock(x, y)
        loss.backward()
        optimizer.step()
    distance = np.sqrt((float(x.data) - 1.0) ** 2 + (float(y.data) - 1.0) ** 2)
    return distance


@pytest.mark.unit
def test_rosenbrock_is_zero_at_the_true_minimum():
    x, y = Tensor(1.0), Tensor(1.0)
    assert rosenbrock(x, y).data == pytest.approx(0.0)


@pytest.mark.oracle
def test_rosenbrock_gradient_matches_torch():
    x, y = Tensor(-1.5), Tensor(2.0)
    loss = rosenbrock(x, y)
    loss.backward()

    tx = torch.tensor(-1.5, requires_grad=True)
    ty = torch.tensor(2.0, requires_grad=True)
    ((1.0 - tx) ** 2 + 100.0 * (ty - tx**2) ** 2).backward()

    assert x.grad == pytest.approx(tx.grad.item())
    assert y.grad == pytest.approx(ty.grad.item())


@pytest.mark.unit
def test_momentum_and_nesterov_reach_near_the_minimum():
    assert _run(Momentum, lr=0.001, momentum=0.9) < 0.01
    assert _run(NesterovMomentum, lr=0.001, momentum=0.9) < 0.01


@pytest.mark.unit
def test_plain_sgd_gets_stuck_far_from_minimum():
    """SGD puro, mesmo orçamento de passos, não chega nem perto — o vale
    curvo de Rosenbrock é o caso clássico que motiva qualquer variante
    além do gradiente cru."""
    assert _run(SGD, lr=0.001) > 0.3


@pytest.mark.unit
def test_momentum_beats_adaptive_methods_on_rosenbrock():
    """O contraponto ao clichê 'Adam é sempre melhor': no vale curvo, onde
    a direção do gradiente é consistente ao longo do caminho, o acúmulo
    de momentum navega melhor que a normalização agressiva por coordenada
    dos métodos adaptativos — com este orçamento de passos, pelo menos."""
    momentum_distance = _run(Momentum, lr=0.001, momentum=0.9)
    adam_distance = _run(Adam, lr=0.01)
    rmsprop_distance = _run(RMSProp, lr=0.01)

    assert momentum_distance < adam_distance
    assert momentum_distance < rmsprop_distance
