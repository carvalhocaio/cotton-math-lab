"""Motor mínimo de autodiff, modo reverso (estilo micrograd, com arrays)."""

import numpy as np

from cotton_math_lab.exceptions import AutodiffError


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

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise AutodiffError(
                "expoente deve ser escalar (int ou float), não outro Tensor "
                "- derivada de x**y em relação a y exigiria log(x), fora "
                "do escopo deste motor mínimo"
            )
        out = Tensor(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            # d(x^n)/dx = n·x^(n-1) — regra do tombo.
            self.grad = self.grad + (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        exponentiated = np.exp(self.data)
        out = Tensor(exponentiated, (self,), "exp")

        def _backward():
            # d(exp(x))/dx = exp(x) — a própria função é a sua derivada.
            self.grad = self.grad + exponentiated * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1.0

    def __rtruediv__(self, other):
        return Tensor(other) * self**-1.0

    def zero_grad(self) -> None:
        """Zera o gradiente acumulado — necessário entre passadas de backward
        que compartilham os mesmos Tensores de entrada (ex: cada linha de
        um Jacobiano), senão o gradiente da próxima passada soma em cima
        do resíduo da anterior."""
        self.grad = np.zeros_like(self.data)

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Propaga gradientes deste nó até todas as folhas do grafo.

        `grad` semeia o gradiente do nó raiz: por padrão, `ones_like` (a
        convenção para saída escalar, onde d(saída)/d(saída) = 1). Para
        Jacobianos, cada linha semeia um vetor one-hot diferente — é assim
        que se extrai "a derivada de só esta saída" de um nó vetorial.
        """
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: "Tensor") -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        self.grad = (
            np.ones_like(self.data)
            if grad is None
            else np.asarray(grad, dtype=np.float64)
        )
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
