import numpy as np
import pytest

from cotton_math_lab.data.hvi import default_spec, generate_bales
from cotton_math_lab.linalg.pca import pca_via_covariance, pca_via_svd

N = 5000
SEED = 2024


def _hvi_sample():
    spec = default_spec()
    return generate_bales(spec, n=N, seed=SEED), spec


def _ill_conditioned_matrix(seed: int = 7, n: int = 300, eps: float = 1e-8):
    """Nearly duplicated columns: cond(X) ~ 2e8 — the same collinearity
    that shows up between correlated HVI features (e.g. uhml/uniformity),
    just amplified to make the effect visible in double precision."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, 3))
    duplicate = base[:, 0] + eps * rng.standard_normal(n)
    return np.column_stack([base[:, 0], duplicate, base[:, 1], base[:, 2]])


@pytest.mark.oracle
def test_explained_variance_matches_numpy_svd():
    bales, _ = _hvi_sample()
    _, explained_variance, _, _ = pca_via_svd(bales)

    centered = bales - bales.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    reference = (singular_values**2) / (len(bales) - 1)

    np.testing.assert_allclose(explained_variance, reference, rtol=1e-8)


@pytest.mark.oracle
def test_components_are_orthonormal():
    bales, _ = _hvi_sample()
    components, _, _, _ = pca_via_svd(bales)
    gram = components.T @ components
    np.testing.assert_allclose(gram, np.eye(components.shape[1]), atol=1e-8)


@pytest.mark.oracle
def test_agrees_with_covariance_route_when_well_conditioned():
    """Both routes agree when there's no extreme collinearity."""
    bales, _ = _hvi_sample()
    _, ev_cov, _, _ = pca_via_covariance(bales)
    _, ev_svd, _, _ = pca_via_svd(bales)
    np.testing.assert_allclose(ev_cov, ev_svd, rtol=1e-6)


@pytest.mark.oracle
def test_svd_route_stays_accurate_on_ill_conditioned_data():
    """The test that proves the trade-off: SVD doesn't lose the
    smallest component even when columns are nearly collinear."""
    matrix = _ill_conditioned_matrix()
    _, explained_variance, _, _ = pca_via_svd(matrix)

    centered = matrix - matrix.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    reference = (singular_values**2) / (len(matrix) - 1)

    relative_error = abs(explained_variance[-1] - reference[-1]) / reference[-1]
    assert relative_error < 1e-6


@pytest.mark.unit
def test_covariance_route_loses_smallest_component_on_ill_conditioned_data():
    """The counterpoint: the same matrix breaks the covariance route.
    Not a bug — it's the squared condition number consuming double
    precision. If this test fails, the doc's trade-off has gone stale."""
    matrix = _ill_conditioned_matrix()
    _, explained_variance, _, _ = pca_via_covariance(matrix)

    centered = matrix - matrix.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    reference = (singular_values**2) / (len(matrix) - 1)

    relative_error = abs(explained_variance[-1] - reference[-1]) / reference[-1]
    assert relative_error > 0.5  # essentially lost the signal
