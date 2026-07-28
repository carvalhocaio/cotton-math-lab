import numpy as np
import pytest

from cotton_math_lab.linalg.eigen import power_iteration


def _symmetric_psd(n: int, seed: int) -> np.ndarray:
    """Matriz simétrica positiva-semidefinida A = M·Mᵀ."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return m @ m.T


@pytest.mark.oracle
def test_dominant_eigenvalue_matches_numpy():
    matrix = _symmetric_psd(6, seed=0)
    eigenvalue, _ = power_iteration(matrix, seed=0)

    reference = np.linalg.eigvalsh(matrix).max()
    assert eigenvalue == pytest.approx(reference, rel=1e-6)


@pytest.mark.oracle
def test_dominant_eigenvector_satisfies_eigen_equation():
    """A·v = λ·v é o que define o par — testamos a definição, não o vetor."""
    matrix = _symmetric_psd(6, seed=1)
    eigenvalue, vector = power_iteration(matrix, seed=0)

    residual = matrix @ vector - eigenvalue * vector
    assert np.linalg.norm(residual) < 1e-4


@pytest.mark.oracle
def test_eigenvector_is_unit_norm():
    matrix = _symmetric_psd(5, seed=2)
    _, vector = power_iteration(matrix, seed=0)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


@pytest.mark.unit
def test_converges_faster_with_larger_spectral_gap():
    """Convergência é geométrica na razão |λ₂/λ₁|: gap maior, menos iterações."""
    wide = np.diag([10.0, 1.0, 0.5])  # razão 0.1
    narrow = np.diag([10.0, 9.0, 0.5])  # razão 0.9

    _, _, iters_wide = power_iteration(wide, seed=0, return_iters=True)
    _, _, iters_narrow = power_iteration(narrow, seed=0, return_iters=True)

    assert iters_wide < iters_narrow


@pytest.mark.unit
def test_raises_on_non_square_matrix():
    from cotton_math_lab.exceptions import LinAlgError

    with pytest.raises(LinAlgError, match="quadrada"):
        power_iteration(np.ones((3, 4)), seed=0)
