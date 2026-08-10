import numpy as np
import pytest

from cotton_math_lab.autodiff.optim import SGD
from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.unit
def test_converges_to_minimum_of_quadratic_bowl():
    """f(x) = (x-3)² tem mínimo único em x=3 - o teste mais simples do
    convergência que existe, sem nenhum dado envolvido."""
    x = Tensor(0.0)
    optimizar = SGD([x], lr=0.1)

    for _ in range(50):
        optimizar.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        optimizar.step()

    assert x.data == pytest.approx(3.0, abs=1e-3)


@pytest.mark.unit
def test_converges_on_toy_linear_regression():
    """y = 2x + 1 exato, sem ruído — w e b devem convergir perto dos
    valores verdadeiros, e a perda deve cair por pelo menos duas ordens
    de grandeza."""
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = 2 * xs + 1.0

    w, b = Tensor(0.0), Tensor(0.0)
    optimizer = SGD([w, b], lr=0.01)

    initial_loss = None
    final_loss = None
    for epoch in range(200):
        optimizer.zero_grad()
        loss = Tensor(0.0)
        for x_val, y_val in zip(xs, ys, strict=True):
            prediction = w * Tensor(x_val) + b
            loss = loss + (prediction - Tensor(y_val)) ** 2
        loss = loss * (1.0 / len(xs))
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.data
        final_loss = loss.data

    assert w.data == pytest.approx(2.0, abs=0.1)
    assert b.data == pytest.approx(1.0, abs=0.15)
    assert final_loss < initial_loss * 0.01


@pytest.mark.unit
def test_zero_grad_resets_all_parameters():
    a, b = Tensor(1.0), Tensor(2.0)
    a.grad = np.array(5.0)
    b.grad = np.array(-3.0)

    optimizer = SGD([a, b], lr=0.1)
    optimizer.zero_grad()

    assert a.grad == pytest.approx(0.0)
    assert b.grad == pytest.approx(0.0)


@pytest.mark.unit
def test_step_moves_parameter_opposite_to_gradient():
    """Definição de gradiente descendente: passo move CONTRA o gradiente."""
    x = Tensor(5.0)
    x.grad = np.array(2.0)  # gradiente positivo -> deveria diminuir x

    SGD([x], lr=0.1).step()

    assert x.data < 5.0
