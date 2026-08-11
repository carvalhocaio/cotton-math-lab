import numpy as np
import pytest
from scipy.optimize import minimize

from cotton_math_lab.autodiff.benchmarks import rosenbrock
from cotton_math_lab.autodiff.hessian import hessian as hessian_fn
from cotton_math_lab.autodiff.second_order import bfgs_minimize, newton_minimize
from cotton_math_lab.autodiff.tensor import Tensor


def _quadratic(xs):
    x0, x1 = xs
    return (x0 - Tensor(3.0)) ** 2 * 2.0 + (x1 - Tensor(-1.0)) ** 2 * 5.0


def _rosenbrock_fn(xs):
    x, y = xs
    return rosenbrock(x, y)


@pytest.mark.unit
def test_newton_converges_in_one_step_on_a_quadratic():
    """Para uma quadrática pura, o passo de Newton é EXATO — a Hessiana é
    constante, e resolver H·Δx=g dá o mínimo direto, sem nenhuma iteração
    extra. Isso não é aproximação: é a definição do método."""
    x0 = np.array([10.0, 10.0])
    x1, iterations = newton_minimize(_quadratic, x0, max_iter=1)

    np.testing.assert_allclose(x1, [3.0, -1.0], atol=1e-8)
    assert iterations == 1


@pytest.mark.unit
def test_newton_converges_in_far_fewer_iterations_than_first_order():
    """No Rosenbrock, onde Momentum precisou de 2000 passos pra chegar a
    ~0.0003 do mínimo, Newton chega a precisão de máquina em menos de 20
    iterações — o preço dessa velocidade é o assunto do próximo teste."""
    x_final, iterations = newton_minimize(_rosenbrock_fn, np.array([-1.5, 2.0]))
    distance = np.linalg.norm(x_final - np.array([1.0, 1.0]))

    assert iterations < 20
    assert distance < 1e-6


@pytest.mark.oracle
def test_bfgs_matches_scipy_result():
    x_bfgs, _ = bfgs_minimize(_rosenbrock_fn, [-1.5, 2.0])

    reference = minimize(
        lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2,
        x0=[-1.5, 2.0],
        method="BFGS",
    )
    np.testing.assert_allclose(x_bfgs, reference.x, atol=1e-4)


@pytest.mark.unit
def test_hessian_construction_cost_scales_linearly_with_dimension():
    """Construir a Hessiana via diferenças finitas custa EXATAMENTE 2n
    avaliações de f — determinístico, não depende de hardware nem de
    tempo de parede. Um passo de primeira ordem custa 1 avaliação,
    sempre, não importa n."""

    def sum_of_squares(xs):
        total = xs[0] * xs[0]
        for x in xs[1:]:
            total = total + x * x
        return total

    for n in (5, 20, 50):
        calls = {"count": 0}

        def counted_fn(xs, calls=calls):
            calls["count"] += 1
            return sum_of_squares(xs)

        hessian_fn(counted_fn, np.zeros(n))
        assert calls["count"] == 2 * n
