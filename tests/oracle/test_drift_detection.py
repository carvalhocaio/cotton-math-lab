import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon

from cotton_math_lab.data.hvi import HVISpec, default_spec, generate_bales
from cotton_math_lab.infotheory.drift import detect_drift, js_divergence_discrete

P = np.array([0.5, 0.35, 0.15])
Q = np.array([0.2, 0.3, 0.5])


@pytest.mark.oracle
def test_js_matches_scipy():
    """scipy.spatial.distance.jensenshannon devolve a RAIZ da divergência
    (a distância JS) — eleva ao quadrado pra comparar com a divergência."""
    ours = js_divergence_discrete(P, Q)
    scipy_distance = jensenshannon(P, Q, base=np.e)
    assert ours == pytest.approx(scipy_distance**2)


@pytest.mark.unit
def test_js_is_symmetric():
    assert js_divergence_discrete(P, Q) == pytest.approx(js_divergence_discrete(Q, P))


@pytest.mark.unit
def test_js_is_zero_for_identical_distributions():
    assert js_divergence_discrete(P, P) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_js_is_bounded_by_log_2():
    """JS nunca ultrapassa log(2) — o caso extremo é distribuições
    completamente disjuntas (nenhum suporte compartilhado)."""
    disjoint_p = np.array([1.0, 0.0, 0.0])
    disjoint_q = np.array([0.0, 0.0, 1.0])
    assert js_divergence_discrete(disjoint_p, disjoint_q) == pytest.approx(np.log(2))


@pytest.mark.unit
def test_detect_drift_distinguishes_sampling_noise_from_real_shift():
    """A aplicação real: duas safras da MESMA distribuição (só ruído
    amostral) não deveriam disparar o detector; uma safra com
    deslocamento real de média deveria disparar, com folga."""
    spec = default_spec()
    baseline = generate_bales(spec, n=1000, seed=1)
    same_distribution = generate_bales(spec, n=1000, seed=2)

    idx = {name: i for i, name in enumerate(spec.features)}
    shifted_means = spec.means.copy()
    shifted_means[idx["micronaire"]] += 0.6
    drifted_spec = HVISpec(
        features=spec.features,
        means=shifted_means,
        stds=spec.stds.copy(),
        correlation=spec.correlation.copy(),
    )
    drifted = generate_bales(drifted_spec, n=1000, seed=3)

    no_drift_flag, no_drift_js = detect_drift(
        baseline[:, idx["micronaire"]], same_distribution[:, idx["micronaire"]]
    )
    real_drift_flag, real_drift_js = detect_drift(
        baseline[:, idx["micronaire"]], drifted[:, idx["micronaire"]]
    )

    assert not no_drift_flag
    assert real_drift_flag
    assert real_drift_js > no_drift_js * 10
