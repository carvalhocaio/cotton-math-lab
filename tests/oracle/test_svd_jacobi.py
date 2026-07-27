import numpy as np
import pytest

from cotton_math_lab.linalg.svd import svd_jacobi_one_sided


def _random_matrix(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n))


@pytest.mark.oracle
def test_singular_values_match_numpy_svd():
    matrix = _random_matrix(10, 5, seed=0)
    _, singular_values, _ = svd_jacobi_one_sided(matrix)

    reference = np.linalg.svd(matrix, compute_uv=False)
    np.testing.assert_allclose(singular_values, reference, rtol=1e-8)


@pytest.mark.oracle
def test_reconstructs_original_matrix():
    matrix = _random_matrix(8, 4, seed=1)
    u, singular_values, v = svd_jacobi_one_sided(matrix)

    reconstructed = u @ np.diag(singular_values) @ v.T
    np.testing.assert_allclose(reconstructed, matrix, atol=1e-8)


@pytest.mark.oracle
def test_right_singular_vectors_are_orthonormal():
    matrix = _random_matrix(10, 6, seed=2)
    _, _, v = svd_jacobi_one_sided(matrix)
    gram = v.T @ v
    np.testing.assert_allclose(gram, np.eye(v.shape[1]), atol=1e-10)


@pytest.mark.oracle
def test_left_singular_vectors_are_orthonormal():
    matrix = _random_matrix(10, 6, seed=3)
    u, _, _ = svd_jacobi_one_sided(matrix)
    gram = u.T @ u
    np.testing.assert_allclose(gram, np.eye(u.shape[1]), atol=1e-10)


@pytest.mark.unit
def test_singular_values_returned_in_descending_order():
    matrix = _random_matrix(10, 5, seed=4)
    _, singular_values, _ = svd_jacobi_one_sided(matrix)
    assert np.all(np.diff(singular_values) <= 0)


@pytest.mark.unit
def test_raises_when_fewer_rows_than_columns():
    from cotton_math_lab.exceptions import LinAlgError

    with pytest.raises(LinAlgError, match="linhas"):
        svd_jacobi_one_sided(_random_matrix(3, 5, seed=5))
