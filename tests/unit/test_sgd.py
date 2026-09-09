import numpy as np
import pytest

from cotton_math_lab.autodiff.optim import SGD
from cotton_math_lab.autodiff.tensor import Tensor


@pytest.mark.unit
def test_converges_to_minimum_of_quadratic_bowl():
    """f(x) = (x-3)² has a unique minimum at x=3 - the simplest possible
    convergence test, with no data involved."""
    x = Tensor(0.0)
    optimizer = SGD([x], lr=0.1)

    for _ in range(50):
        optimizer.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        optimizer.step()

    assert x.data == pytest.approx(3.0, abs=1e-3)


@pytest.mark.unit
def test_converges_on_toy_linear_regression():
    """y = 2x + 1 exactly, no noise — w and b should converge close to
    the true values, and the loss should drop by at least two orders of
    magnitude."""
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
    """Definition of gradient descent: the step moves AGAINST the gradient."""
    x = Tensor(5.0)
    x.grad = np.array(2.0)  # positive gradient -> x should decrease

    SGD([x], lr=0.1).step()

    assert x.data < 5.0
