import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.stats.beta_binomial import beta_binomial_posterior, posterior_mean


@pytest.mark.oracle
def test_posterior_mean_matches_grid_search():
    """Valida contra maximização numérica direta da posterior — prior ×
    verossimilhança binomial, integrada numa grade fina, sem assumir a
    fórmula fechada em nenhum momento."""
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
    """Prior quase-flat (α,β → 0) -> posterior dominada pelos dados, e a
    média posterior converge para k/n, o MLE de uma binomial."""
    k, n = 6, 40
    alpha_post, beta_post = beta_binomial_posterior(k, n, 1e-6, 1e-6)
    assert posterior_mean(alpha_post, beta_post) == pytest.approx(k / n, abs=1e-4)


@pytest.mark.unit
def test_strong_prior_dominates_with_little_data():
    """Um prior concentrado (α+β grande) com poucos dados observados
    (n pequeno) deveria puxar a média posterior de volta pra perto da
    média do prior, quase ignorando os dados."""
    strong_alpha, strong_beta = 200.0, 800.0  # prior concentrado perto de 0.2
    prior_mean = strong_alpha / (strong_alpha + strong_beta)

    k, n = 1, 3  # 1 fardo fora de spec em 3 — MLE=0.33, bem longe do prior
    alpha_post, beta_post = beta_binomial_posterior(k, n, strong_alpha, strong_beta)

    assert posterior_mean(alpha_post, beta_post) == pytest.approx(prior_mean, abs=0.02)


@pytest.mark.unit
def test_sequential_updates_equal_single_batch_update():
    """Propriedade central de conjugação: atualizar em dois lotes (4/20,
    depois 2/20) deve dar a MESMA posterior que atualizar uma vez com os
    dados combinados (6/40) — a ordem de chegada da evidência não importa."""
    prior = (2.0, 18.0)

    intermediate = beta_binomial_posterior(4, 20, *prior)
    sequential = beta_binomial_posterior(2, 20, *intermediate)

    batch = beta_binomial_posterior(6, 40, *prior)

    assert sequential == pytest.approx(batch)
