import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from cotton_math_lab.infotheory.discrete import (
    cross_entropy_discrete,
    entropy_discrete,
    kl_divergence_discrete,
)

# distribution of bale quality grades: premium, standard, off-spec
P = np.array([0.5, 0.35, 0.15])
Q = np.array([0.4, 0.4, 0.2])


@pytest.mark.oracle
def test_entropy_matches_scipy():
    assert entropy_discrete(P) == pytest.approx(scipy_entropy(P))


@pytest.mark.oracle
def test_kl_divergence_matches_scipy():
    """scipy.stats.entropy(p, q) computes KL(p||q) when both arguments
    are passed — no separate library is needed for KL."""
    assert kl_divergence_discrete(P, Q) == pytest.approx(scipy_entropy(P, Q))


@pytest.mark.unit
def test_cross_entropy_decomposes_into_entropy_plus_kl():
    """H(p,q) = H(p) + D_KL(p||q) — the identity that connects the
    cycle's three concepts, tested directly, not assumed."""
    cross = cross_entropy_discrete(P, Q)
    decomposed = entropy_discrete(P) + kl_divergence_discrete(P, Q)
    assert cross == pytest.approx(decomposed)


@pytest.mark.unit
def test_kl_is_zero_between_identical_distributions():
    assert kl_divergence_discrete(P, P) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_kl_satisfies_gibbs_inequality():
    """D_KL(p||q) ≥ 0 always, for any pair of distributions — tested on
    5000 random pairs via the simplex (Dirichlet), not just the domain
    example."""
    rng = np.random.default_rng(0)
    for _ in range(5000):
        a = rng.dirichlet(np.ones(4))
        b = rng.dirichlet(np.ones(4))
        assert kl_divergence_discrete(a, b) >= -1e-10


@pytest.mark.unit
def test_uniform_distribution_maximizes_entropy():
    """Among 3000 random distributions over 5 categories, none exceeds
    the entropy of the uniform (log 5) — the uniform is the global
    maximum, not just a good candidate."""
    n = 5
    uniform_entropy = entropy_discrete(np.ones(n) / n)

    rng = np.random.default_rng(1)
    for _ in range(3000):
        p = rng.dirichlet(np.ones(n))
        assert entropy_discrete(p) <= uniform_entropy + 1e-10

    assert uniform_entropy == pytest.approx(np.log(n))
