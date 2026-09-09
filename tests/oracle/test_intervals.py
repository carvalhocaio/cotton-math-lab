import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.stats.beta_binomial import beta_binomial_posterior
from cotton_math_lab.stats.intervals import (
    beta_credible_interval,
    wilson_score_interval,
)

TRUE_P = 0.15
N_OBS = 40


@pytest.mark.oracle
def test_wilson_matches_score_test_inversion():
    """Validates the closed form against the definition: direct
    inversion of the score test on a fine grid of p0, without assuming
    the Wilson formula."""
    k, n = 6, 40
    z_crit = stats.norm.ppf(0.975)
    p_hat = k / n

    p0_grid = np.linspace(1e-6, 1 - 1e-6, 200_000)
    z_stat = (p_hat - p0_grid) / np.sqrt(p0_grid * (1 - p0_grid) / n)
    accepted = np.abs(z_stat) <= z_crit
    grid_interval = (p0_grid[accepted].min(), p0_grid[accepted].max())

    closed_form = wilson_score_interval(k, n)
    assert closed_form[0] == pytest.approx(grid_interval[0], abs=1e-4)
    assert closed_form[1] == pytest.approx(grid_interval[1], abs=1e-4)


@pytest.mark.unit
def test_credible_interval_captures_exact_confidence_mass():
    """The interval must contain EXACTLY `confidence` of the mass under
    the posterior - tests the interval's definition, not the formula."""
    alpha, beta = 8.0, 34.0
    lower, upper = beta_credible_interval(alpha, beta, confidence=0.95)
    mass = stats.beta.cdf(upper, alpha, beta) - stats.beta.cdf(lower, alpha, beta)
    assert mass == pytest.approx(0.95, abs=1e-6)


@pytest.mark.unit
def test_intervals_diverge_with_an_informative_prior():
    """Proof of divergence in a single concrete example: with a strong
    prior (centered at 0.20, BEFORE seeing the data), the credible
    interval and the confidence interval - which uses no prior - don't
    coincide."""
    k, n = 6, 40
    ci = wilson_score_interval(k, n)

    strong_prior = (40.0, 160.0)
    alpha_post, beta_post = beta_binomial_posterior(k, n, *strong_prior)
    credible = beta_credible_interval(alpha_post, beta_post)

    assert abs(ci[0] - credible[0]) > 0.03
    assert abs(ci[1] - credible[1]) > 0.03


@pytest.mark.slow
def test_ci_coverage_is_robust_regardless_of_prior():
    """The confidence interval uses no prior at all - its coverage
    should stay close to 95% always, and that's what validates the
    frequentist guarantee."""
    rng = np.random.default_rng(2024)
    n_experiments = 1500
    covered = 0

    for _ in range(n_experiments):
        k_obs = rng.binomial(N_OBS, TRUE_P)
        if k_obs == 0:
            lower, upper = 0.0, wilson_score_interval(1, N_OBS)[1]
        else:
            lower, upper = wilson_score_interval(k_obs, N_OBS)
        if lower <= TRUE_P <= upper:
            covered += 1

    assert 0.90 < covered / n_experiments < 0.99


@pytest.mark.slow
def test_credible_interval_coverage_degrades_with_mismatched_prior():
    """The module's central point: a strong prior MISMATCHED with the
    truth (centered at 0.20, truth is 0.15) makes the credible
    interval's coverage drop well below 95% — the frequentist guarantee
    simply doesn't exist for a credible interval under a wrong prior,
    even though the Bayesian interpretation remains valid (95%
    posterior belief, given THAT prior)."""
    rng = np.random.default_rng(2024)
    n_experiments = 1500
    mismatched_prior = (40.0, 160.0)
    covered = 0

    for _ in range(n_experiments):
        k_obs = rng.binomial(N_OBS, TRUE_P)
        alpha_post, beta_post = beta_binomial_posterior(k_obs, N_OBS, *mismatched_prior)
        lower, upper = beta_credible_interval(alpha_post, beta_post)
        if lower <= TRUE_P <= upper:
            covered += 1

    coverage = covered / n_experiments
    assert coverage < 0.85  # well below the nominal 95%


@pytest.mark.slow
def test_credible_interval_coverage_recovers_with_weak_prior():
    """With a weak (nearly non-informative) prior, the credible
    interval recovers frequentist coverage close to 95% — the two
    framings converge numerically when the prior carries little
    weight."""
    rng = np.random.default_rng(2024)
    n_experiments = 1500
    covered = 0

    for _ in range(n_experiments):
        k_obs = rng.binomial(N_OBS, TRUE_P)
        alpha_post, beta_post = beta_binomial_posterior(k_obs, N_OBS, 1.0, 1.0)
        lower, upper = beta_credible_interval(alpha_post, beta_post)
        if lower <= TRUE_P <= upper:
            covered += 1

    assert 0.90 < covered / n_experiments < 0.99
