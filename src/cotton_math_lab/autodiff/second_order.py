"""Newton e quasi-Newton (BFGS) — reaproveitam gradient() e hessian() do
Módulo 2 diretamente. Nenhuma matemática nova, só uma aplicação nova das
mesmas peças."""

import numpy as np

from cotton_math_lab.autodiff.gradient import gradient
from cotton_math_lab.autodiff.hessian import hessian as hessian_fn
from cotton_math_lab.autodiff.tensor import Tensor


def newton_minimize(
    f, x0: np.ndarray, max_iter: int = 50, tol: float = 1e-10
) -> tuple[np.ndarray, int]:
    """Método de Newton: a cada passo, resolve H·Δx = -g e anda Δx.

    Para uma quadrática, a Hessiana é constante e o passo é EXATO — uma
    única iteração encontra o mínimo, porque a aproximação quadrática de
    segunda ordem que o método usa não é uma aproximação nesse caso, é a
    função exata. Fora de quadráticas, cada passo minimiza a aproximação
    quadrática local, e a convergência tende a ser rapidíssima perto do
    mínimo — ao custo de montar e resolver um sistema linear n×n a cada
    iteração.
    """
    x = np.asarray(x0, dtype=np.float64).copy()

    for iteration in range(1, max_iter + 1):
        g = gradient(f, x)
        if np.linalg.norm(g) < tol:
            return x, iteration - 1
        H = hessian_fn(f, x)
        x = x - np.linalg.solve(H, g)

    return x, max_iter


def bfgs_minimize(
    f, x0: np.ndarray, max_iter: int = 200, tol: float = 1e-8
) -> tuple[np.ndarray, int]:
    """Quasi-Newton BFGS: aproxima a INVERSA da Hessiana a partir só de
    gradientes, sem nunca montar a Hessiana verdadeira.

    A cada passo, atualiza a aproximação H⁻¹ usando a equação secante — a
    mudança observada no gradiente informa sobre a curvatura, sem
    precisar de segunda derivada explícita. Custa O(n²) por passo, bem
    menos que os O(n) gradientes mais O(n³) de resolver um sistema que
    Newton puro exige a cada iteração — ao preço de precisar de mais
    iterações pra convergir com a mesma qualidade.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    n = len(x)
    h_inv = np.eye(n)
    g = gradient(f, x)
    iteration = 0

    for iteration in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            break

        direction = -h_inv @ g

        step = 1.0
        f_x = float(f([Tensor(v) for v in x]).data)
        while step > 1e-12:
            x_new = x + step * direction
            f_new = float(f([Tensor(v) for v in x_new]).data)
            if f_new < f_x:
                break
            step *= 0.5
        x_new = x + step * direction
        g_new = gradient(f, x_new)

        s = x_new - x
        y = g_new - g
        sy = s @ y
        if (
            sy > 1e-10
        ):  # só atualiza com curvatura positiva (mantém H⁻¹ definida positiva)
            rho = 1.0 / sy
            identity = np.eye(n)
            h_inv = (identity - rho * np.outer(s, y)) @ h_inv @ (
                identity - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)

        x, g = x_new, g_new

    return x, iteration
