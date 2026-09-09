import numpy as np
import pytest
from sklearn.metrics import mutual_info_score

from cotton_math_lab.data.hvi import default_spec, generate_bales
from cotton_math_lab.infotheory.mutual_information import mutual_information_binned


@pytest.mark.oracle
def test_mi_matches_sklearn_with_consistent_binning():
    """Uses the SAME bin edges for both calculations — without this, a
    small discretization difference already breaks the exact
    comparison."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = 0.7 * x + rng.standard_normal(2000) * 0.5

    bins = 15
    _, xedges, yedges = np.histogram2d(x, y, bins=bins)
    mi_ours = mutual_information_binned(x, y, bins=bins)

    x_labels = np.digitize(x, xedges[1:-1])
    y_labels = np.digitize(y, yedges[1:-1])
    mi_sklearn = mutual_info_score(x_labels, y_labels)

    assert mi_ours == pytest.approx(mi_sklearn, abs=1e-10)


@pytest.mark.unit
def test_mi_is_near_zero_for_independent_variables():
    """Like any estimator with a finite sample, MI has a small upward
    bias even under true independence — hence the tolerance not being
    exactly zero, the same kind of small-sample bias seen in Module 3."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    assert mutual_information_binned(x, y, bins=15) < 0.1


@pytest.mark.unit
def test_mi_detects_nonlinear_dependence_that_correlation_misses():
    """The cycle's central argument: Y=X² with X symmetric around zero
    has Pearson correlation ≈ 0 (the dependence is perfectly real, but
    nonlinear, and Pearson only sees linear) — MI has no such blind
    spot."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 5000)
    y = x**2

    pearson_correlation = np.corrcoef(x, y)[0, 1]
    mi = mutual_information_binned(x, y, bins=20)

    assert abs(pearson_correlation) < 0.05  # Pearson sees nothing
    assert mi > 1.0  # MI sees a strong, real dependence


@pytest.mark.unit
def test_mi_ranks_correlated_hvi_features_above_uncorrelated_ones():
    """Real application: in Module 0's HVI data, MI should be much
    larger between features correlated by construction (uhml,
    uniformity — correlation 0.55) than between features independent by
    construction (micronaire, rd — correlation 0)."""
    spec = default_spec()
    bales = generate_bales(spec, n=3000, seed=2024)
    idx = {name: i for i, name in enumerate(spec.features)}

    mi_correlated = mutual_information_binned(
        bales[:, idx["uhml"]], bales[:, idx["uniformity"]]
    )
    mi_uncorrelated = mutual_information_binned(
        bales[:, idx["micronaire"]], bales[:, idx["rd"]]
    )

    assert mi_correlated > mi_uncorrelated * 3
