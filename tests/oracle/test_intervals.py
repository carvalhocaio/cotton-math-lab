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
    """Valida a forma fechada contra a definição: inversão direta do teste
    de score numa grade fina de p0, sem assumir a fórmula do Wilson."""
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
    """O intervalo deve conter EXATAMENTE `confidence` de massa sob a
    posterior - testa a definição do intervalo, não a fórmula."""
    alpha, beta = 8.0, 34.0
    lower, upper = beta_credible_interval(alpha, beta, confidence=0.95)
    mass = stats.beta.cdf(upper, alpha, beta) - stats.beta.cdf(lower, alpha, beta)
    assert mass == pytest.approx(0.95, abs=1e-6)


@pytest.mark.unit
def test_intervals_diverge_with_an_informative_prior():
    """Prova de divergência num único exemplo concreto: com um prior forte
    (centrado em 0.20, ANTES de ver os dados), o intervalo de credibilidade
    e o IC - que não usa prior nenhum - não coincidem."""
    k, n = 6, 40
    ci = wilson_score_interval(k, n)

    strong_prior = (40.0, 160.0)
    alpha_post, beta_post = beta_binomial_posterior(k, n, *strong_prior)
    credible = beta_credible_interval(alpha_post, beta_post)

    assert abs(ci[0] - credible[0]) > 0.03
    assert abs(ci[1] - credible[1]) > 0.03


@pytest.mark.slow
def test_ci_coverage_is_robust_regardless_of_prior():
    """O IC não usa prior nenhum - sua cobertura deveria ficar perto de
    95% sempre, e é isso que valida a garantia frequentista."""
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
    """O ponto central do módulo: um prior forte e DESCASADO com a
    verdade (centrado em 0.20, verdade é 0.15) faz a cobertura do
    intervalo de credibilidade cair bem abaixo de 95% — a garantia
    frequentista simplesmente não existe pra intervalo de credibilidade
    sob um prior errado, mesmo que a interpretação bayesiana continue
    válida (95% de crença posterior, dado ESSE prior)."""
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
    assert coverage < 0.85  # bem baixo dos 95% nominais


@pytest.mark.slow
def test_credible_interval_coverage_recovers_with_weak_prior():
    """Com um prior fraco (quase não-informativo), o intervalo de
    credibilidade recupera cobertura frequentista próxima de 95% — os
    dois enquadramentos convergem numericamente quando o prior não pesa."""
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
