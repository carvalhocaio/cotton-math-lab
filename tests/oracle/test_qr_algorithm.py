import numpy as np
import pytest

from cotton_math_lab.linalg.eigen import qr_algorithm


def _symmetric_matrix(eigenvalues: np.ndarray, seed: int) -> np.ndarray:
    """Constrói A = Q·diag(eigenvalues)·Qᵀ com Q ortogonal aleatória."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((len(eigenvalues), len(eigenvalues)))
    q, _ = np.linalg.qr(m)
    return q @ np.diag(eigenvalues) @ q.T


@pytest.mark.oracle
def test_recovers_full_spectrum_matches_numpy():
    matrix = _symmetric_matrix(np.array([10.0, 7.0, 3.2, 0.7, 0.3, 0.1]), seed=0)
    eigenvalues, _ = qr_algorithm(matrix)

    reference = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    ordered = eigenvalues[np.argsort(-np.abs(eigenvalues))]
    np.testing.assert_allclose(ordered, reference, atol=1e-6)


@pytest.mark.oracle
def test_recovered_eigenvectors_satisfy_eigen_equation():
    matrix = _symmetric_matrix(np.array([8.0, -5.0, 3.0, -1.0, 0.4]), seed=1)
    eigenvalues, eigenvectors = qr_algorithm(matrix)

    residual = matrix @ eigenvectors - eigenvectors @ np.diag(eigenvalues)
    assert np.linalg.norm(residual) < 1e-6


@pytest.mark.oracle
def test_eigenvectors_are_orthonormal():
    matrix = _symmetric_matrix(np.array([6.0, 4.0, 2.0, 1.0]), seed=2)
    _, eigenvectors = qr_algorithm(matrix)
    gram = eigenvectors.T @ eigenvectors
    np.testing.assert_allclose(gram, np.eye(4), atol=1e-8)


@pytest.mark.oracle
def test_no_orthogonality_degradation_across_deflation_like_use():
    """O ponto que a deflação não tinha: precisão não cai nos últimos pares."""
    matrix = _symmetric_matrix(np.linspace(20.0, 0.1, 8), seed=3)
    _, eigenvectors = qr_algorithm(matrix)

    gram = eigenvectors.T @ eigenvectors
    off_diagonal = gram - np.eye(8)
    assert np.abs(off_diagonal).max() < 1e-8


@pytest.mark.unit
def test_narrow_spectral_gap_needs_far_more_iterations():
    """Mesmo mecanismo da power iteration: taxa governada por |λ_{k+1}/λ_k|."""
    wide = _symmetric_matrix(np.array([10.0, 1.0, 0.5, 0.1]), seed=4)
    narrow = _symmetric_matrix(np.array([10.0, 9.999, 5.0, 1.0]), seed=4)

    _, _, iters_wide = qr_algorithm(wide, return_iters=True, max_iter=3000)
    _, _, iters_narrow = qr_algorithm(narrow, return_iters=True, max_iter=3000)

    assert iters_wide < iters_narrow


@pytest.mark.unit
def test_raises_on_non_symmetric_matrix():
    from cotton_math_lab.exceptions import LinAlgError

    rng = np.random.default_rng(2)
    matrix = rng.standard_normal((4, 4))
    with pytest.raises(LinAlgError, match="simétrica"):
        qr_algorithm(matrix)


@pytest.mark.unit
def test_raises_on_non_square_matrix():
    from cotton_math_lab.exceptions import LinAlgError

    with pytest.raises(LinAlgError, match="quadrada"):
        qr_algorithm(np.ones((3, 4)))
