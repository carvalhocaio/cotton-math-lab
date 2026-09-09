import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.stats.beta_binomial import beta_binomial_posterior, posterior_mean


@pytest.mark.oracle
def test_posterior_mean_matches_grid_search():
    """Validates against direct numerical maximization of the posterior
    — prior × binomial likelihood, integrated on a fine grid, without
    ever assuming the closed form."""
    k, n = 6, 40
    alpha_prior, beta_prior = 2.0, 18.0

    p_grid = np.linspace(1e-6, 1 - 1e-6, 50_000)
    log_prior = stats.beta.logpdf(p_grid, alpha_prior, beta_prior)
    log_likelihood = k * np.log(p_grid) + (n - k) * np.log(1 - p_grid)
    log_posterior = log_prior + log_likelihood
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= np.trapezoid(posterior, p_grid)
    mean_grid = np.trapezoid(p_grid * posterior, p_grid)

    alpha_post, beta_post = beta_binomial_posterior(k, n, alpha_prior, beta_prior)
    assert posterior_mean(alpha_post, beta_post) == pytest.approx(mean_grid, abs=1e-6)


@pytest.mark.oracle
def test_posterior_mean_matches_scipy_beta():
    alpha_post, beta_post = beta_binomial_posterior(6, 40, 2.0, 18.0)
    reference = stats.beta.mean(alpha_post, beta_post)
    assert posterior_mean(alpha_post, beta_post) == pytest.approx(reference)


@pytest.mark.unit
def test_weak_prior_converges_to_mle():
    """Nearly-flat prior (α,β → 0) -> posterior dominated by the data,
    and the posterior mean converges to k/n, the MLE of a binomial."""
    k, n = 6, 40
    alpha_post, beta_post = beta_binomial_posterior(k, n, 1e-6, 1e-6)
    assert posterior_mean(alpha_post, beta_post) == pytest.approx(k / n, abs=1e-4)


@pytest.mark.unit
def test_strong_prior_dominates_with_little_data():
    """A concentrated prior (α+β large) with little observed data
    (small n) should pull the posterior mean back close to the prior
    mean, nearly ignoring the data."""
    strong_alpha, strong_beta = 200.0, 800.0  # prior concentrated near 0.2
    prior_mean = strong_alpha / (strong_alpha + strong_beta)

    k, n = 1, 3  # 1 bale out of spec out of 3 — MLE=0.33, far from the prior
    alpha_post, beta_post = beta_binomial_posterior(k, n, strong_alpha, strong_beta)

    assert posterior_mean(alpha_post, beta_post) == pytest.approx(prior_mean, abs=0.02)


@pytest.mark.unit
def test_sequential_updates_equal_single_batch_update():
    """The core conjugation property: updating in two batches (4/20,
    then 2/20) should give the SAME posterior as updating once with the
    combined data (6/40) — the order evidence arrives in doesn't matter."""
    prior = (2.0, 18.0)

    intermediate = beta_binomial_posterior(4, 20, *prior)
    sequential = beta_binomial_posterior(2, 20, *intermediate)

    batch = beta_binomial_posterior(6, 40, *prior)

    assert sequential == pytest.approx(batch)
