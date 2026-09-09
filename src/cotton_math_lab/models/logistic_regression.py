"""Logistic regression built entirely from Module 2's pieces.

No line here comes from numpy doing the math ready-made: `sigmoid` is a
composition of exp and division, the loss is a composition of log and
sum, and the entire training loop runs through `Tensor.backward()` and
`SGD` — the same pieces validated against torch, finite differences, and
each other throughout the whole module.
"""

import numpy as np

from cotton_math_lab.autodiff.optim import SGD
from cotton_math_lab.autodiff.tensor import Tensor


def sigmoid(t: Tensor) -> Tensor:
    return 1.0 / (1.0 + (-t).exp())


def binary_cross_entropy(prediction: Tensor, target: float) -> Tensor:
    return -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log())


class LogisticRegression:
    """Linear binary classifier, trained by SGD on top of `Tensor`."""

    def __init__(self, n_features: int):
        self.weights = Tensor(np.zeros(n_features))
        self.bias = Tensor(0.0)

    def parameters(self) -> list[Tensor]:
        return [self.weights, self.bias]

    def forward(self, x: np.ndarray) -> Tensor:
        z = (self.weights * Tensor(x)).sum() + self.bias
        return sigmoid(z)

    def predict(self, x: np.ndarray) -> float:
        return 1.0 if self.forward(x).data > 0.5 else 0.0

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        lr: float = 0.15,
        epochs: int = 120,
    ) -> list[float]:
        optimizer = SGD(self.parameters(), lr=lr)
        history = []

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = Tensor(0.0)
            for x_row, y_value in zip(x_train, y_train, strict=True):
                prediction = self.forward(x_row)
                loss = loss + binary_cross_entropy(prediction, y_value)
            loss = loss * (1.0 / len(x_train))

            loss.backward()
            optimizer.step()
            history.append(float(loss.data))

        return history
