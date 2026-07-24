import numpy as np
import pytest

from cotton_math_lab.data.hvi import default_spec, generate_bales


@pytest.mark.unit
def test_generates_requested_number_of_bales():
    spec = default_spec()
    bales = generate_bales(spec, n=100, seed=42)
    assert bales.shape == (100, len(spec.features))


@pytest.mark.unit
def test_same_seed_produces_identical_bales():
    spec = default_spec()
    first = generate_bales(spec, n=50, seed=7)
    second = generate_bales(spec, n=50, seed=7)
    np.testing.assert_array_equal(first, second)


@pytest.mark.unit
def test_different_seeds_produce_different_bales():
    spec = default_spec()
    first = generate_bales(spec, n=50, seed=7)
    second = generate_bales(spec, n=50, seed=8)
    assert not np.array_equal(first, second)
