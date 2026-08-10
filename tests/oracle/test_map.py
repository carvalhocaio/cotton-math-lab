import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.stats.map import map_normal_mean

TRUE_MU, TRUE_SIGMA = 4.3, 0.4


@pytest.mark.slow
def test_map_matches_grid_search_posterior():
    """Valida a forma fechada contra busca em grade na posterior real —
    nenhuma fórmula assumida, só maximização direta de prior × verossimilhança."""
    rng = np.random.default_rng(0)
    data = rng.normal(TRUE_MU, TRUE_SIGMA, 5)
    prior_mean, prior_std = 4.0, 0.3

    mu_grid = np.linspace(2.0, 6.0, 20_000)
    log_prior = stats.norm.logpdf(mu_grid, prior_mean, prior_std)
    log_likelihood = np.array(
        [stats.norm.logpdf(m, data, TRUE_SIGMA).sum() for m in mu_grid]
    )
    mu_map_grid = mu_grid[np.argmax(log_prior + log_likelihood)]

    mu_map_closed = map_normal_mean(data, TRUE_SIGMA, prior_mean, prior_std)
    assert mu_map_closed == pytest.approx(mu_map_grid, abs=1e-3)


@pytest.mark.unit
def test_map_converges_to_mle_with_weak_prior():
    """Prior quase não-informativo (variância enorme) -> MAP ≈ MLE."""
    rng = np.random.default_rng(1)
    data = rng.normal(TRUE_MU, TRUE_SIGMA, 10)

    mle = data.mean()
    map_estimate = map_normal_mean(data, TRUE_SIGMA, prior_mean=0.0, prior_std=1e6)
    assert map_estimate == pytest.approx(mle, abs=1e-4)


@pytest.mark.unit
def test_map_converges_to_prior_with_very_informative_prior():
    """Prior muito confiante (variância minúscula) -> MAP ≈ prior_mean,
    não importa o que os dados digam."""
    rng = np.random.default_rng(2)
    data = rng.normal(TRUE_MU, TRUE_SIGMA, 10)

    prior_mean = 4.0
    map_estimate = map_normal_mean(data, TRUE_SIGMA, prior_mean, prior_std=1e-6)
    assert map_estimate == pytest.approx(prior_mean, abs=1e-4)


@pytest.mark.unit
def test_map_reduces_error_with_small_samples_and_reasonable_prior():
    """O trade-off central do módulo: com poucos dados, um prior só
    aproximadamente certo (4.2, não os 4.3 verdadeiros) ainda reduz o erro
    quadrático médio pela metade em relação ao MLE cru."""
    rng = np.random.default_rng(2024)
    n_trials, n_small = 5000, 4

    mle_errors = np.zeros(n_trials)
    map_errors = np.zeros(n_trials)
    for i in range(n_trials):
        sample = rng.normal(TRUE_MU, TRUE_SIGMA, n_small)
        mle_errors[i] = (sample.mean() - TRUE_MU) ** 2
        map_errors[i] = (
            map_normal_mean(sample, TRUE_SIGMA, prior_mean=4.2, prior_std=0.3) - TRUE_MU
        ) ** 2

    reduction = 1.0 - map_errors.mean() / mle_errors.mean()
    assert reduction > 0.2  # medido ~50%; margem generosa contra ruído


@pytest.mark.unit
def test_map_advantage_vanishes_with_large_samples():
    """A mesma comparação, mas com n=200 — os dados devem dominar o prior,
    e a vantagem de MAP sobre MLE deve praticamente desaparecer."""
    rng = np.random.default_rng(2024)
    n_trials, n_large = 5000, 200

    mle_errors = np.zeros(n_trials)
    map_errors = np.zeros(n_trials)
    for i in range(n_trials):
        sample = rng.normal(TRUE_MU, TRUE_SIGMA, n_large)
        mle_errors[i] = (sample.mean() - TRUE_MU) ** 2
        map_errors[i] = (
            map_normal_mean(sample, TRUE_SIGMA, prior_mean=4.2, prior_std=0.3) - TRUE_MU
        ) ** 2

    ratio = map_errors.mean() / mle_errors.mean()
    assert ratio == pytest.approx(1.0, abs=0.1)
