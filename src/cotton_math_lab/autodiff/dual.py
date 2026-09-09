"""Dual numbers: x + ε·x', with ε² = 0.

Each arithmetic operation propagates the derivative together with the
value, in a single forward pass — unlike reverse mode, which first
records the graph (forward) and only afterward propagates the gradient
(backward). Forward mode needs no graph and no backward phase: the
derivative already comes out ready in the dual part, in the same pass
that computes the value.
"""

import numpy as np


class Dual:
    """A dual number: `.real` is the value, `.dual` is the accumulated derivative."""

    __slots__ = ("real", "dual")

    def __init__(self, real: float, dual: float = 0.0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        # the product rule comes for free: (a+εa')(b+εb') = ab + ε(ab'+a'b),
        # since ε² = 0 discards the cross term a'b'ε².
        return Dual(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real,
        )

    __rmul__ = __mul__

    def __sub__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __pow__(self, exponent: float):
        return Dual(
            self.real**exponent,
            exponent * self.real ** (exponent - 1) * self.dual,
        )

    def exp(self):
        value = np.exp(self.real)
        return Dual(value, value * self.dual)
