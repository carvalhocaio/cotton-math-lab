import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.stats.bootstrap import bootstrap_confidence_interval

TRUE_MU, TRUE_SIGMA = 4.3, 0.4


@pytest.mark.oracle
def test_bootstrap_ci_matches_scipy_for_the_mean():
    rng = np.random.default_rng(2024)
    data = rng.normal(TRUE_MU, TRUE_SIGMA, 60)

    lower, upper = bootstrap_confidence_interval(
        data, np.mean, n_resamples=5000, seed=1
    )

    reference = stats.bootstrap(
        (data,),
        np.mean,
        confidence_level=0.95,
        n_resamples=5000,
        method="percentile",
        random_state=np.random.default_rng(1),
    )
    assert lower == pytest.approx(reference.confidence_interval.low, abs=1e-9)
    assert upper == pytest.approx(reference.confidence_interval.high, abs=1e-9)


@pytest.mark.oracle
def test_bootstrap_works_for_median_without_closed_form():
    """A estatística que prova o argumento do ciclo: mediana de uma
    distribuição assimétrica não tem fórmula fechada simples de erro-padrão
    — bootstrap não precisa de uma."""
    rng = np.random.default_rng(7)
    data = rng.exponential(scale=2.0, size=80)

    lower, upper = bootstrap_confidence_interval(
        data, np.median, n_resamples=5000, seed=3
    )

    reference = stats.bootstrap(
        (data,),
        np.median,
        confidence_level=0.95,
        n_resamples=5000,
        method="percentile",
        random_state=np.random.default_rng(3),
    )
    assert lower == pytest.approx(reference.confidence_interval.low, abs=1e-9)
    assert upper == pytest.approx(reference.confidence_interval.high, abs=1e-9)


@pytest.mark.slow
def test_bootstrap_ci_has_approximately_correct_coverage():
    """O teste que importa de verdade: um IC de 95% deveria conter a
    verdade em ~95% de repetições independentes do experimento inteiro —
    não só parecer razoável numa única amostra."""
    rng = np.random.default_rng(2024)
    n_experiments, n_resamples, sample_size = 1000, 500, 50

    covered = 0
    for i in range(n_experiments):
        sample = rng.normal(TRUE_MU, TRUE_SIGMA, sample_size)
        lower, upper = bootstrap_confidence_interval(
            sample, np.mean, n_resamples=n_resamples, seed=i
        )
        if lower <= TRUE_MU <= upper:
            covered += 1

    coverage = covered / n_experiments
    assert 0.88 < coverage < 0.98


@pytest.mark.unit
def test_bootstrap_is_deterministic_given_seed():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 30)

    first = bootstrap_confidence_interval(data, np.mean, seed=42)
    second = bootstrap_confidence_interval(data, np.mean, seed=42)
    assert first == second
