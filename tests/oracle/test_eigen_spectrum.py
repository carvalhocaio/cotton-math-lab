import numpy as np
import pytest

from cotton_math_lab.linalg.eigen import eigen_spectrum


def _symmetric_psd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return m @ m.T


@pytest.mark.oracle
def test_full_spectrum_matches_numpy():
    matrix = _symmetric_psd(6, seed=0)
    eigenvalues, _ = eigen_spectrum(matrix, seed=0)

    reference = np.sort(np.linalg.eigvalsh(matrix))[::-1]
    np.testing.assert_allclose(eigenvalues, reference, rtol=1e-6)


@pytest.mark.oracle
def test_eigenvalues_returned_in_descending_order():
    matrix = _symmetric_psd(6, seed=3)
    eigenvalues, _ = eigen_spectrum(matrix, seed=0)
    assert np.all(np.diff(eigenvalues) <= 0)


@pytest.mark.oracle
def test_recovered_eigenvectors_are_orthonormal():
    """Autovetores de matriz simétrica formam base ortonormal: VᵀV = I."""
    matrix = _symmetric_psd(6, seed=4)
    _, vectors = eigen_spectrum(matrix, seed=0)

    gram = vectors.T @ vectors
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-5)


@pytest.mark.oracle
def test_eigendecomposition_reconstructs_matrix():
    """A = V·Λ·Vᵀ — o teste mais forte: o espectro inteiro remonta A."""
    matrix = _symmetric_psd(5, seed=5)
    eigenvalues, vectors = eigen_spectrum(matrix, seed=0)

    reconstructed = vectors @ np.diag(eigenvalues) @ vectors.T
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-4)


@pytest.mark.unit
def test_partial_spectrum_returns_only_k_components():
    matrix = _symmetric_psd(6, seed=6)
    eigenvalues, vectors = eigen_spectrum(matrix, k=2, seed=0)
    assert eigenvalues.shape == (2,)
    assert vectors.shape == (6, 2)
