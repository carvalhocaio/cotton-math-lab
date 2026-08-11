import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from cotton_math_lab.infotheory.discrete import (
    cross_entropy_discrete,
    entropy_discrete,
    kl_divergence_discrete,
)

# distribuição de faixas de qualidade de fardos: premium, standard, off-spec
P = np.array([0.5, 0.35, 0.15])
Q = np.array([0.4, 0.4, 0.2])


@pytest.mark.oracle
def test_entropy_matches_scipy():
    assert entropy_discrete(P) == pytest.approx(scipy_entropy(P))


@pytest.mark.oracle
def test_kl_divergence_matches_scipy():
    """scipy.stats.entropy(p, q) calcula KL(p||q) quando os dois argumentos
    são passados — não precisa de biblioteca separada pra KL."""
    assert kl_divergence_discrete(P, Q) == pytest.approx(scipy_entropy(P, Q))


@pytest.mark.unit
def test_cross_entropy_decomposes_into_entropy_plus_kl():
    """H(p,q) = H(p) + D_KL(p||q) — a identidade que conecta os três
    conceitos do ciclo, testada diretamente, não assumida."""
    cross = cross_entropy_discrete(P, Q)
    decomposed = entropy_discrete(P) + kl_divergence_discrete(P, Q)
    assert cross == pytest.approx(decomposed)


@pytest.mark.unit
def test_kl_is_zero_between_identical_distributions():
    assert kl_divergence_discrete(P, P) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_kl_satisfies_gibbs_inequality():
    """D_KL(p||q) ≥ 0 sempre, pra qualquer par de distribuições — testado
    em 5000 pares aleatórios via simplex (Dirichlet), não só no exemplo
    do domínio."""
    rng = np.random.default_rng(0)
    for _ in range(5000):
        a = rng.dirichlet(np.ones(4))
        b = rng.dirichlet(np.ones(4))
        assert kl_divergence_discrete(a, b) >= -1e-10


@pytest.mark.unit
def test_uniform_distribution_maximizes_entropy():
    """Entre 3000 distribuições aleatórias sobre 5 categorias, nenhuma
    supera a entropia da uniforme (log 5) — a uniforme é o máximo global,
    não só um bom candidato."""
    n = 5
    uniform_entropy = entropy_discrete(np.ones(n) / n)

    rng = np.random.default_rng(1)
    for _ in range(3000):
        p = rng.dirichlet(np.ones(n))
        assert entropy_discrete(p) <= uniform_entropy + 1e-10

    assert uniform_entropy == pytest.approx(np.log(n))
