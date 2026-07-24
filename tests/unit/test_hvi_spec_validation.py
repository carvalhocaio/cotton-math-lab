import numpy as np
import pytest

from cotton_math_lab.data.hvi import FEATURES, HVISpec, default_spec
from cotton_math_lab.exceptions import InvalidSpecError


def _spec_with(**overrides) -> HVISpec:
    """Constrói um spec a partir do default, com campos substituídos."""
    base = default_spec()
    fields = {
        "features": base.features,
        "means": base.means.copy(),
        "stds": base.stds.copy(),
        "correlation": base.correlation.copy(),
    }
    return HVISpec(**{**fields, **overrides})


@pytest.mark.unit
def test_rejects_non_positive_std():
    stds = default_spec().stds.copy()
    stds[3] = -2.5
    with pytest.raises(InvalidSpecError, match="desvio"):
        _spec_with(stds=stds)


@pytest.mark.unit
def test_rejects_asymmetric_correlation():
    corr = default_spec().correlation.copy()
    corr[1, 2] = 0.9  # sem espelhar em [2, 1]
    with pytest.raises(InvalidSpecError, match="simétrica"):
        _spec_with(correlation=corr)


@pytest.mark.unit
def test_rejects_correlation_with_non_unit_diagonal():
    corr = default_spec().correlation.copy()
    corr[0, 0] = 0.8
    with pytest.raises(InvalidSpecError, match="diagonal"):
        _spec_with(correlation=corr)


@pytest.mark.unit
def test_rejects_non_positive_definite_correlation():
    idx = {name: i for i, name in enumerate(FEATURES)}
    corr = default_spec().correlation.copy()
    contradictory = {
        ("uhml", "uniformity"): 0.95,
        ("uhml", "strength"): 0.95,
        ("uniformity", "strength"): -0.95,
    }
    for (a, b), value in contradictory.items():
        corr[idx[a], idx[b]] = value
        corr[idx[b], idx[a]] = value

    with pytest.raises(InvalidSpecError, match="positiva-definida"):
        _spec_with(correlation=corr)


@pytest.mark.unit
def test_rejects_mismatched_dimensions():
    with pytest.raises(InvalidSpecError, match="dimens"):
        _spec_with(means=np.array([4.3, 29.0]))


@pytest.mark.unit
def test_default_spec_is_valid():
    assert default_spec() is not None
