import numpy as np
import pytest
from triton.language import standard

from cotton_math_lab.data.hvi import default_spec, generate_bales

N_LARGE = 20_000
SEED = 2024
Z_TOL = 4.0  # ~1 em 16 mil por feature; margem medida foi 2.6

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


@pytest.mark.unit
def test_recovers_population_means():
    """Erro da média amostral deve caber em Z_TOL erros-padrão (σ/√n)."""
    spec = default_spec()
    bales = generate_bales(spec, n=N_LARGE, seed=SEED)

    standard_error = spec.stds / np.sqrt(N_LARGE)
    z_scores = np.abs(bales.mean(axis=0) - spec.means) / standard_error

    assert z_scores.max() < Z_TOL, dict(zip(spec.features, z_scores, strict=True))


@pytest.mark.unit
def test_recovers_population_stds():
    """Erro-padrão do desvio amostral é σ/√(2n)."""
    spec = default_spec()
    bales = generate_bales(spec, n=N_LARGE, seed=SEED)

    standard_error = spec.stds / np.sqrt(2 * N_LARGE)
    z_scores = np.abs(bales.std(axis=0, ddof=1) - spec.stds) / standard_error

    assert z_scores.max() < Z_TOL, dict(zip(spec.features, z_scores, strict=True))


@pytest.mark.unit
def test_recovers_population_correlation():
    """Compara em espaço-z de Fisher, onde o erro-padrão é 1/√(n-3)."""
    spec = default_spec()
    bales = generate_bales(spec, n=N_LARGE, seed=SEED)

    empirical = np.corrcoef(bales, rowvar=False)
    upper = np.triu_indices(len(spec.features), k=1)

    z_error = np.abs(np.arctanh(empirical[upper]) - np.arctanh(spec.correlation[upper]))
    z_scores = z_error * np.sqrt(N_LARGE - 3)

    assert z_scores.max() < Z_TOL, z_scores.max()
