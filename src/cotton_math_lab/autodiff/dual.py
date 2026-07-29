"""Números duais: x + ε·x', com ε² = 0.

Cada operação aritmética propaga a derivada com o valor, num único
passe para frente — ao contrário do modo reverso, que primeiro grava o
grafo (forward) e só depois propaga gradiente (backward). Forward mode não
precisa de grafo, nem de fase de backward: a derivada já sai pronta na
parte dual no mesmo passe que calcula o valor.
"""

import numpy as np


class Dual:
    """Um número dual: `.real` é o valor, `.dual` é a derivada acumulada."""

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
        # regra do produto sai de graça: (a+εa')(b+εb') = ab + ε(ab'+a'b),
        # já que ε² = 0 descarta o termo cruzado a'b'ε².
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
