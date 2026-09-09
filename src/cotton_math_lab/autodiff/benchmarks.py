"""Classic test surfaces for comparing optimizers."""

from cotton_math_lab.autodiff.tensor import Tensor


def rosenbrock(x: Tensor, y: Tensor, a: float = 1.0, b: float = 100.0) -> Tensor:
    """f(x,y) = (a-x)² + b(y-x²)² — global minimum at (a, a²), value 0.

    The valley around the minimum is narrow and CURVED (follows the
    parabola y=x²), not aligned with any axis — unlike the
    ill-conditioned quadratic from earlier cycles, whose principal axes
    coincide with x and y. It's this curvature, not just scale, that
    makes Rosenbrock hard: even a method that handles different
    curvature directions well (RMSProp, Adam) may not navigate it better
    than a method that just accumulates a consistent direction
    (Momentum), because the challenge here was never unequal scale
    between axes.
    """
    return (Tensor(a) - x) ** 2 + Tensor(b) * (y - x**2) ** 2


def ill_conditioned_quadratic(x: Tensor, y: Tensor, ratio: float = 10.0) -> Tensor:
    """f(x,y) = x² + ratio·y² — the surface used in the Momentum,
    RMSProp, and Adam cycles. Curvature `ratio`× larger in y than in x;
    unlike Rosenbrock, the principal axes coincide with x and y, so the
    challenge here is purely one of SCALE, not curved trajectory.
    """
    return x**2 + Tensor(ratio) * y**2
