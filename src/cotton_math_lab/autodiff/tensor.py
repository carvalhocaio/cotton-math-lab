"""Motor mínimo de autodiff, modo reverso (estilo micrograd, com arrays)."""

import numpy as np


class Tensor:
    """Nó de um grafo computacional com diferenciação automática reversa.

    Cada operação (`+`, `*`, ...) cria um novo Tensor que guarda, em
    `_backward`, a regra local de como distribuir o gradiente de saída
    para seus operandos. `backward()` percorre o grafo em ordem
    topológica reversa e aplica essas regras em cascata — é a regra da
    cadeia, uma aplicação local por vez.
    """

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1 — o gradiente de saída passa
            # direto para os dois operandos.
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a — regra do produto.
            self.grad = self.grad + other.data * out.grad
            other.grad = other.grad + self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def backward(self):
        """Propaga gradientes deste nó até todas as folhas do grafo."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: "Tensor") -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
