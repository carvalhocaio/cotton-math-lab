import pytest
import torch

from cotton_math_lab.autodiff.optim import SGD, Momentum, NesterovMomentum
from cotton_math_lab.autodiff.tensor import Tensor


def _run(optimizer_cls, steps=40, **kwargs) -> float:
    x, y = Tensor(3.0), Tensor(3.0)
    optimizer = optimizer_cls([x, y], **kwargs)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = x**2 + y**2 * 10.0
        loss.backward()
        optimizer.step()
    return float(x.data**2 + y.data**2 * 10.0)


@pytest.mark.oracle
def test_nesterov_trajectory_matches_torch():
    x, y = Tensor(3.0), Tensor(3.0)
    optimizer = NesterovMomentum([x, y], lr=0.05, momentum=0.9)

    tx = torch.tensor(3.0, requires_grad=True)
    ty = torch.tensor(3.0, requires_grad=True)
    torch_optimizer = torch.optim.SGD([tx, ty], lr=0.05, momentum=0.9, nesterov=True)

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
def test_nesterov_is_more_stable_than_classic_momentum():
    """At the lr where classic Momentum already gets WORSE than plain
    SGD (found in the previous cycle), Nesterov stays stable and beats
    both — the "look-ahead" correction shifts the stability boundary,
    it's not just theoretical elegance."""
    lr = 0.03
    sgd_loss = _run(SGD, lr=lr)
    momentum_loss = _run(Momentum, lr=lr, momentum=0.9)
    nesterov_loss = _run(NesterovMomentum, lr=lr, momentum=0.9)

    assert momentum_loss > sgd_loss  # reconfirms the previous cycle's instability
    assert nesterov_loss < sgd_loss  # Nesterov doesn't suffer the same instability


@pytest.mark.unit
def test_nesterov_beats_classic_momentum_when_well_tuned():
    lr = 0.02
    momentum_loss = _run(Momentum, lr=lr, momentum=0.9)
    nesterov_loss = _run(NesterovMomentum, lr=lr, momentum=0.9)

    assert nesterov_loss < momentum_loss
