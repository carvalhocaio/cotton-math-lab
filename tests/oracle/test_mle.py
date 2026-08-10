import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.autodiff.optim import SGD
from cotton_math_lab.autodiff.tensor import Tensor
from cotton_math_lab.stats.mle import mle_normal

SEED = 2024
TRUE_MU, TRUE_SIGMA = 4.3, 0.4


def _micronaire_sample(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(TRUE_MU, TRUE_SIGMA, n)


@pytest.mark.oracle
def test_mle_normal_matches_scipy():
    data = _micronaire_sample(150)
    mu_hat, sigma_hat = mle_normal(data)

    mu_ref, sigma_ref = stats.norm.fit(data)
    assert mu_hat == pytest.approx(mu_ref)
    assert sigma_hat == pytest.approx(sigma_ref)


@pytest.mark.oracle
def test_mle_normal_matches_gradient_based_optimization():
    """Valida a forma fechada contra o motor de autodiff do Módulo 2:
    maximiza a log-verossimilhança numericamente via SGD, partindo de um
    chute deliberadamente ruim (μ=0), e confere que os dois caminhos —
    álgebra fechada e otimização por gradiente — chegam no mesmo lugar."""
    data = _micronaire_sample(150)
    mu_closed, sigma_closed = mle_normal(data)

    mu = Tensor(0.0)
    log_sigma = Tensor(0.0)  # sigma = exp(log_sigma) > 0 sempre, por construção
    optimizer = SGD([mu, log_sigma], lr=0.05)
    n = len(data)

    for _ in range(400):
        optimizer.zero_grad()
        sigma = log_sigma.exp()
        nll = Tensor(0.0)
        for x in data:
            diff = Tensor(x) - mu
            nll = nll + log_sigma + (diff**2) * 0.5 * (sigma**-2)
        nll = nll * (1.0 / n)
        nll.backward()
        optimizer.step()

    assert float(mu.data) == pytest.approx(mu_closed, abs=1e-4)
    assert float(np.exp(log_sigma.data)) == pytest.approx(sigma_closed, abs=1e-4)


@pytest.mark.unit
def test_mle_mean_is_unbiased():
    """μ_MLE = média amostral, sempre não-enviesado — E[μ̂] = μ verdadeiro."""
    rng = np.random.default_rng(7)
    means = np.array(
        [mle_normal(rng.normal(TRUE_MU, TRUE_SIGMA, 8))[0] for _ in range(3000)]
    )
    assert means.mean() == pytest.approx(TRUE_MU, abs=0.05)


@pytest.mark.unit
def test_mle_variance_is_biased_low():
    """σ²_MLE subestima a variância populacional na razão exata (n-1)/n —
    a 'correção de Bessel' que MLE não aplica, e todo outro estimador aplica."""
    rng = np.random.default_rng(7)
    n_small = 8
    variances = np.array(
        [
            mle_normal(rng.normal(TRUE_MU, TRUE_SIGMA, n_small))[1] ** 2
            for _ in range(3000)
        ]
    )

    observed_ratio = variances.mean() / TRUE_SIGMA**2
    expected_ratio = (n_small - 1) / n_small
    assert observed_ratio == pytest.approx(expected_ratio, abs=0.05)
