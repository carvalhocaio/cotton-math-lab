import numpy as np
import pytest

from cotton_math_lab.data.hvi import default_spec, generate_bales
from cotton_math_lab.linalg.pca import pca_via_covariance

N = 5_000
SEED = 2_024


def _hvi_sample():
    spec = default_spec()
    return generate_bales(spec, n=N, seed=SEED), spec


@pytest.mark.oracle
def test_explained_variance_matches_numpy_svd():
    bales, _ = _hvi_sample()
    _, explained_variance, _, _ = pca_via_covariance(bales)

    centered = bales - bales.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    reference = (singular_values**2) / (len(bales) - 1)

    np.testing.assert_allclose(
        explained_variance, reference[: len(explained_variance)], rtol=1e-6
    )


@pytest.mark.oracle
def test_components_are_orthonormal():
    bales, _ = _hvi_sample()
    components, _, _, _ = pca_via_covariance(bales)
    gram = components.T @ components
    np.testing.assert_allclose(gram, np.eye(components.shape[1]), atol=1e-8)


@pytest.mark.oracle
def test_component_satisfies_eigen_equation():
    """Tests the definition (Cov·v = λ·v) — avoids the vector's sign ambiguity."""
    bales, _ = _hvi_sample()
    components, explained_variance, mean, scale = pca_via_covariance(bales)

    centered = (bales - mean) / scale
    covariance = (centered.T @ centered) / (len(bales) - 1)

    residual = covariance @ components - components @ np.diag(explained_variance)
    assert np.linalg.norm(residual) < 1e-6


@pytest.mark.oracle
def test_standardized_variance_sums_to_number_of_features():
    """Trace of the correlation matrix = p, diagonal all 1s. Invariant,
    no oracle needed."""
    bales, spec = _hvi_sample()
    _, explained_variance, _, _ = pca_via_covariance(bales, standardize=True)
    assert explained_variance.sum() == pytest.approx(len(spec.features), rel=1e-3)


@pytest.mark.oracle
def test_raw_covariance_pc1_is_dominated_by_largest_variance_feature():
    """The gotcha: without standardizing, PC1 becomes almost just the
    largest-scale feature."""
    bales, spec = _hvi_sample()
    components, _, _, _ = pca_via_covariance(bales, standardize=False)

    pc1_top = spec.features[np.argmax(np.abs(components[:, 0]))]
    highest_variance_feature = spec.features[np.argmax(bales.var(axis=0, ddof=1))]

    assert pc1_top == highest_variance_feature


@pytest.mark.unit
def test_standardizing_shifts_pc1_away_from_raw_scale_dominance():
    """The fix: standardized, PC1 is no longer hostage to the unit of measurement."""
    bales, spec = _hvi_sample()
    raw_components, _, _, _ = pca_via_covariance(bales, standardize=False)
    std_components, _, _, _ = pca_via_covariance(bales, standardize=True)

    raw_top = spec.features[np.argmax(np.abs(raw_components[:, 0]))]
    std_top = spec.features[np.argmax(np.abs(std_components[:, 0]))]

    assert raw_top != std_top
