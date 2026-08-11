"""Superfícies de teste clássicas pra comparar otimizadores."""

from cotton_math_lab.autodiff.tensor import Tensor


def rosenbrock(x: Tensor, y: Tensor, a: float = 1.0, b: float = 100.0) -> Tensor:
    """f(x,y) = (a-x)² + b(y-x²)² — mínimo global em (a, a²), valor 0.

    O vale em torno do mínimo é estreito e CURVO (segue a parábola y=x²),
    não alinhado com nenhum eixo — diferente da quadrática mal-condicionada
    dos ciclos anteriores, cujos eixos principais coincidem com x e y. É
    essa curvatura, não só a escala, que torna Rosenbrock difícil: mesmo
    um método que lida bem com direções de curvatura diferentes (RMSProp,
    Adam) pode não navegar melhor que um método que só acumula direção
    consistente (Momentum), porque o desafio aqui é seguir uma trajetória
    curva, não equalizar duas escalas fixas.
    """
    return (Tensor(a) - x) ** 2 + Tensor(b) * (y - x**2) ** 2


def ill_conditioned_quadratic(x: Tensor, y: Tensor, ratio: float = 10.0) -> Tensor:
    """f(x,y) = x² + ratio·y² — a superfície usada nos ciclos de Momentum,
    RMSProp e Adam. Curvatura `ratio`× maior em y que em x; ao contrário
    de Rosenbrock, os eixos principais coincidem com x e y, então o
    desafio aqui é puramente de ESCALA, não de trajetória curva.
    """
    return x**2 + Tensor(ratio) * y**2
