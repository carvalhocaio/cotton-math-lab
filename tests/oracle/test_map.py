import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.stats.map import map_normal_mean

TRUE_MU, TRUE_SIGMA = 4.3, 0.4


@pytest.mark.slow
def test_map_matches_grid_search_posterior():
    """Validates the closed form against a grid search on the real
    posterior — no formula assumed, just direct maximization of
    prior × likelihood."""
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
    """Nearly non-informative prior (huge variance) -> MAP ≈ MLE."""
    rng = np.random.default_rng(1)
    data = rng.normal(TRUE_MU, TRUE_SIGMA, 10)

    mle = data.mean()
    map_estimate = map_normal_mean(data, TRUE_SIGMA, prior_mean=0.0, prior_std=1e6)
    assert map_estimate == pytest.approx(mle, abs=1e-4)


@pytest.mark.unit
def test_map_converges_to_prior_with_very_informative_prior():
    """Very confident prior (tiny variance) -> MAP ≈ prior_mean, no
    matter what the data says."""
    rng = np.random.default_rng(2)
    data = rng.normal(TRUE_MU, TRUE_SIGMA, 10)

    prior_mean = 4.0
    map_estimate = map_normal_mean(data, TRUE_SIGMA, prior_mean, prior_std=1e-6)
    assert map_estimate == pytest.approx(prior_mean, abs=1e-4)


@pytest.mark.unit
def test_map_reduces_error_with_small_samples_and_reasonable_prior():
    """The module's central trade-off: with little data, a prior that's
    only approximately correct (4.2, not the true 4.3) still cuts the
    mean squared error in half relative to the raw MLE."""
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
    assert reduction > 0.2  # measured ~50%; generous margin against noise


@pytest.mark.unit
def test_map_advantage_vanishes_with_large_samples():
    """The same comparison, but with n=200 — the data should dominate
    the prior, and MAP's advantage over MLE should nearly vanish."""
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
