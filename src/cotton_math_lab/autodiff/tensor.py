"""Minimal reverse-mode autodiff engine (micrograd-style, with arrays)."""

import numpy as np

from cotton_math_lab.exceptions import AutodiffError


class Tensor:
    """Node of a computational graph with reverse automatic differentiation.

    Each operation (`+`, `*`, ...) creates a new Tensor that stores, in
    `_backward`, the local rule for how to distribute the output
    gradient to its operands. `backward()` walks the graph in reverse
    topological order and applies these rules in cascade — it's the
    chain rule, one local application at a time.
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
            # d(a+b)/da = 1, d(a+b)/db = 1 — the output gradient passes
            # straight through to both operands.
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a — the product rule.
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
                "exponent must be a scalar (int or float), not another "
                "Tensor - the derivative of x**y with respect to y would "
                "require log(x), outside the scope of this minimal engine"
            )
        out = Tensor(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            # d(x^n)/dx = n·x^(n-1) — the power rule.
            self.grad = self.grad + (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        exponentiated = np.exp(self.data)
        out = Tensor(exponentiated, (self,), "exp")

        def _backward():
            # d(exp(x))/dx = exp(x) — the function is its own derivative.
            self.grad = self.grad + exponentiated * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1.0

    def __rtruediv__(self, other):
        return Tensor(other) * self**-1.0

    def zero_grad(self) -> None:
        """Zeroes the accumulated gradient — necessary between backward
        passes that share the same input Tensors (e.g. each row of a
        Jacobian), otherwise the next pass's gradient would add on top
        of the previous pass's residual."""
        self.grad = np.zeros_like(self.data)

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Propagates gradients from this node to every leaf of the graph.

        `grad` seeds the root node's gradient: by default, `ones_like`
        (the convention for a scalar output, where d(output)/d(output) =
        1). For Jacobians, each row seeds a different one-hot vector —
        that's how you extract "the derivative of just this output" from
        a vector-valued node.
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

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            # d(log(x))/dx = 1/x
            self.grad = self.grad + (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), "sum")

        def _backward():
            # d(sum(x))/dxᵢ = 1 for every i — the gradient spreads back
            # equally to each element that went into the sum.
            self.grad = self.grad + np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out
