import numpy as np
import pytest

from cotton_math_lab.linalg.qr import qr_gram_schmidt, qr_householder


def _random_matrix(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n))


def _ill_conditioned_matrix() -> np.ndarray:
    """Nearly parallel columns: the classic failure case for classic GS."""
    return np.array(
        [
            [1.0, 1.0, 1.0],
            [1e-7, 0.0, 0.0],
            [0.0, 1e-7, 0.0],
            [0.0, 0.0, 1e-7],
        ]
    )


@pytest.mark.oracle
@pytest.mark.parametrize("qr_impl", [qr_gram_schmidt, qr_householder])
def test_reconstructs_original_matrix(qr_impl):
    matrix = _random_matrix(6, 4, seed=0)
    q, r = qr_impl(matrix)
    np.testing.assert_allclose(q @ r, matrix, atol=1e-10)


@pytest.mark.oracle
@pytest.mark.parametrize("qr_impl", [qr_gram_schmidt, qr_householder])
def test_r_is_upper_triangular(qr_impl):
    matrix = _random_matrix(6, 4, seed=1)
    _, r = qr_impl(matrix)
    lower = np.tril(r, k=-1)
    np.testing.assert_allclose(lower, np.zeros_like(lower), atol=1e-10)


@pytest.mark.oracle
@pytest.mark.parametrize("qr_impl", [qr_gram_schmidt, qr_householder])
def test_q_columns_are_orthonormal_when_well_conditioned(qr_impl):
    matrix = _random_matrix(6, 4, seed=2)
    q, _ = qr_impl(matrix)
    gram = q.T @ q
    np.testing.assert_allclose(gram, np.eye(q.shape[1]), atol=1e-10)


@pytest.mark.oracle
def test_householder_stays_orthogonal_on_ill_conditioned_matrix():
    """The test that proves the trade-off: Householder doesn't degrade near zero."""
    matrix = _ill_conditioned_matrix()
    q, _ = qr_householder(matrix)
    gram = q.T @ q
    orthogonality_error = np.linalg.norm(gram - np.eye(q.shape[1]))
    assert orthogonality_error < 1e-8


@pytest.mark.unit
def test_classical_gram_schmidt_loses_orthogonality_on_ill_conditioned_matrix():
    """Characterizes the known failure — not a bug, it's the reason the
    module exists."""
    matrix = _ill_conditioned_matrix()
    q, _ = qr_gram_schmidt(matrix)
    gram = q.T @ q
    orthogonality_error = np.linalg.norm(gram - np.eye(q.shape[1]))
    assert orthogonality_error > 1e-4  # genuinely loses orthogonality
