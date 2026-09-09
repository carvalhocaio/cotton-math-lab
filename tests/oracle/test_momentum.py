import pytest
import torch

from cotton_math_lab.autodiff.optim import SGD, Momentum
from cotton_math_lab.autodiff.tensor import Tensor


def _ill_conditioned_quadratic_loss(x: Tensor, y: Tensor) -> Tensor:
    """f(x,y) = x² + 10y² — curvature 2 in x, 20 in y: the same
    geometric signature that motivates momentum (and, later, Adam) to
    exist."""
    return x**2 + y**2 * 10.0


@pytest.mark.oracle
def test_momentum_trajectory_matches_torch():
    x, y = Tensor(3.0), Tensor(3.0)
    optimizer = Momentum([x, y], lr=0.05, momentum=0.9)

    tx = torch.tensor(3.0, requires_grad=True)
    ty = torch.tensor(3.0, requires_grad=True)
    torch_optimizer = torch.optim.SGD([tx, ty], lr=0.05, momentum=0.9)

    for _ in range(15):
        optimizer.zero_grad()
        loss = _ill_conditioned_quadratic_loss(x, y)
        loss.backward()
        optimizer.step()

        torch_optimizer.zero_grad()
        torch_loss = tx**2 + ty**2 * 10.0
        torch_loss.backward()
        torch_optimizer.step()

    assert x.data == pytest.approx(tx.item(), abs=1e-5)
    assert y.data == pytest.approx(ty.item(), abs=1e-5)


@pytest.mark.unit
def test_momentum_beats_sgd_when_learning_rate_is_well_tuned():
    """Within a well-tuned lr window, Momentum converges much faster
    than plain SGD on the same ill-conditioned surface."""

    def run(optimizer_cls, **kwargs):
        x, y = Tensor(3.0), Tensor(3.0)
        optimizer = optimizer_cls([x, y], **kwargs)
        for _ in range(40):
            optimizer.zero_grad()
            loss = _ill_conditioned_quadratic_loss(x, y)
            loss.backward()
            optimizer.step()
        return float(x.data**2 + y.data**2 * 10.0)

    final_sgd = run(SGD, lr=0.02)
    final_momentum = run(Momentum, lr=0.02, momentum=0.9)

    assert final_momentum < final_sgd * 0.5


@pytest.mark.unit
def test_momentum_can_be_less_stable_then_sgd_at_higher_lr():
    """The honest counterpoint: near the stability limit, Momentum can
    end up WORSE than plain SGD - it amplifies the effective step, so
    an lr that SGD still tolerates can already destabilize Momentum."""

    def run(optimizer_cls, **kwargs):
        x, y = Tensor(3.0), Tensor(3.0)
        optimizer = optimizer_cls([x, y], **kwargs)
        for _ in range(40):
            optimizer.zero_grad()
            loss = _ill_conditioned_quadratic_loss(x, y)
            loss.backward()
            optimizer.step()
        return float(x.data**2 + y.data**2 * 10.0)

    final_sgd = run(SGD, lr=0.03)
    final_momentum = run(Momentum, lr=0.03, momentum=0.9)

    assert final_momentum > final_sgd
