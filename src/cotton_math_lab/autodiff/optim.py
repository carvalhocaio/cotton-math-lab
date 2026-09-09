"""Stochastic gradient descent - minimal, just what the capstone needs.

Momentum, Nesterov, RMSProp, Adam/AdamW are left for Module 4
(Optimization), where each variant gets its own trade-off comparison.
Here, just the crudest form of the method: one step, one signal.
"""

import numpy as np

from cotton_math_lab.autodiff.tensor import Tensor


class SGD:
    """Updates each parameter in the direction opposite its gradient.

    θ ← θ - lr·∇θ. It's the simplest possible gradient descent step: no
    memory of previous steps, no per-parameter rate adaptation — just
    the local gradient signal, scaled by `lr`.
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data = parameter.data - self.lr * parameter.grad

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class Momentum:
    """SGD with classic (heavy ball) momentum.

    v ← momentum·v + ∇θ
    θ ← θ - lr·v

    The current gradient accumulates into a "velocity vector" that
    carries exponentially decaying memory of past gradients. In a
    direction where the gradient consistently points the same way, v
    grows and accelerates progress; in a direction where the gradient
    flips sign every step, the contributions tend to partially cancel
    out over time. But note: v has no automatic upper bound — the
    effective step can reach ~lr/(1-momentum), much larger than plain
    lr. That's the very mechanism that accelerates convergence in a
    smooth direction and that can destabilize it in a high-curvature
    direction, depending on how close `lr` is to the stability limit.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocity, strict=True):
            velocity[...] = self.momentum * velocity + parameter.grad
            parameter.data = parameter.data - self.lr * velocity

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class NesterovMomentum:
    """Nesterov momentum — corrects the gradient BEFORE applying the
    step, using the velocity that's already accumulated.

    v ← momentum·v + ∇θ
    θ ← θ - lr·(∇θ + momentum·v)

    The classic Nesterov formulation computes the gradient at a
    "future" position (θ - lr·momentum·v), evaluating the function there
    before taking the step — a second forward pass per iteration.
    PyTorch (and this implementation) uses an equivalent algebraic
    reformulation that avoids this second evaluation: it adds
    momentum·v to the CURRENT gradient before scaling by lr, arriving at
    the same destination without the extra cost. In practice, this
    look-ahead correction reduces the overshoot that makes classic
    Momentum oscillate on ill-conditioned surfaces — it shifts the
    stability boundary, it's not just a cosmetic variation.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocity, strict=True):
            velocity[...] = self.momentum * velocity + parameter.grad
            update = parameter.grad + self.momentum * velocity
            parameter.data = parameter.data - self.lr * update

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class RMSProp:
    """Adapts the step size per parameter via a moving average of the
    squared gradient — changes axis compared to Momentum: instead of
    smoothing the DIRECTION of the step, it rescales its MAGNITUDE,
    parameter by parameter.

    v ← α·v + (1-α)·(∇θ)²
    θ ← θ - lr·∇θ / (√v + ε)

    Parameters with historically large gradients (high curvature, like
    y in the test surface) accumulate a large v and get a SMALLER
    effective step; parameters with historically small gradients get a
    LARGER effective step. The practical result: directions with very
    different curvature end up moving at similar rates, without needing
    to hunt for an `lr` that works for both simultaneously — the problem
    Momentum, in the previous cycle, didn't solve on its own.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_average = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for parameter, square_avg in zip(
            self.parameters, self.square_average, strict=True
        ):
            square_avg[...] = self.alpha * square_avg + (1 - self.alpha) * (
                parameter.grad**2
            )
            parameter.data = parameter.data - self.lr * parameter.grad / (
                np.sqrt(square_avg) + self.eps
            )

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class Adam:
    """Combines Momentum (moving average of the gradient) with RMSProp
    (moving average of the squared gradient), plus bias correction —
    necessary because m₀=v₀=0 biases both averages toward zero in the
    first steps, and the two are biased in DIFFERENT proportions
    (governed by β₁ and β₂ respectively), so the imbalance between them
    distorts the step size if left uncorrected.

    m ← β₁·m + (1-β₁)·∇θ
    v ← β₂·v + (1-β₂)·(∇θ)²
    m̂ ← m / (1-β₁ᵗ),  v̂ ← v / (1-β₂ᵗ)
    θ ← θ - lr·m̂ / (√v̂ + ε)

    At t=1, the correction exactly cancels the introduced bias factor:
    m̂ = ∇θ and v̂ = (∇θ)², so m̂/√v̂ = sign(∇θ) — the first step is
    always ±lr, regardless of the initial gradient's magnitude.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for parameter, m, v in zip(self.parameters, self.m, self.v, strict=True):
            m[...] = self.beta1 * m + (1 - self.beta1) * parameter.grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (parameter.grad**2)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            parameter.data = parameter.data - self.lr * m_hat / (
                np.sqrt(v_hat) + self.eps
            )

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class AdamW(Adam):
    """Adam with DECOUPLED weight decay — not 'Adam + L2 on the gradient'.

    θ ← θ - lr·(m̂/(√v̂+ε) + weight_decay·θ)

    The difference is where the decay term enters. Adding
    weight_decay·θ to the GRADIENT (the naive approach) turns that term
    into part of m and v — and so it ends up rescaled by Adam's
    per-parameter adaptation (1/√v̂). Parameters with a history of large
    gradients (large v) end up receiving LESS effective decay;
    parameters with a small history receive MORE — the regularization
    strength ends up depending on each parameter's training history, not
    just on the `weight_decay` you chose.

    Here, the `weight_decay·θ` term is added AFTER the division by √v̂ —
    it never enters the moving averages, never gets rescaled by the
    adaptation. The same nominal decay fraction applies equally to every
    parameter, regardless of how much it has already moved.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def step(self) -> None:
        self.t += 1
        for parameter, m, v in zip(self.parameters, self.m, self.v, strict=True):
            m[...] = self.beta1 * m + (1 - self.beta1) * parameter.grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (parameter.grad**2)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            adaptive_update = m_hat / (np.sqrt(v_hat) + self.eps)
            parameter.data = parameter.data - self.lr * (
                adaptive_update + self.weight_decay * parameter.data
            )
