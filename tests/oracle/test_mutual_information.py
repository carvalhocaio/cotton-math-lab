import numpy as np
import pytest
from sklearn.metrics import mutual_info_score

from cotton_math_lab.data.hvi import default_spec, generate_bales
from cotton_math_lab.infotheory.mutual_information import mutual_information_binned


@pytest.mark.oracle
def test_mi_matches_sklearn_with_consistent_binning():
    """Usa os MESMOS bin edges pros dois cálculos — sem isso, uma pequena
    diferença de discretização já quebra a comparação exata."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = 0.7 * x + rng.standard_normal(2000) * 0.5

    bins = 15
    _, xedges, yedges = np.histogram2d(x, y, bins=bins)
    mi_ours = mutual_information_binned(x, y, bins=bins)

    x_labels = np.digitize(x, xedges[1:-1])
    y_labels = np.digitize(y, yedges[1:-1])
    mi_sklearn = mutual_info_score(x_labels, y_labels)

    assert mi_ours == pytest.approx(mi_sklearn, abs=1e-10)


@pytest.mark.unit
def test_mi_is_near_zero_for_independent_variables():
    """Como qualquer estimador com amostra finita, MI tem viés pequeno
    pra cima mesmo sob independência real — daí a tolerância não ser
    zero exato, é o mesmo tipo de viés de amostra pequena do Módulo 3."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    assert mutual_information_binned(x, y, bins=15) < 0.1


@pytest.mark.unit
def test_mi_detects_nonlinear_dependence_that_correlation_misses():
    """O argumento central do ciclo: Y=X² com X simétrico em torno de
    zero tem correlação de Pearson ≈ 0 (a dependência é perfeitamente
    real, mas não-linear, e Pearson só vê linear) — MI não tem esse
    ponto cego."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 5000)
    y = x**2

    pearson_correlation = np.corrcoef(x, y)[0, 1]
    mi = mutual_information_binned(x, y, bins=20)

    assert abs(pearson_correlation) < 0.05  # Pearson não vê nada
    assert mi > 1.0  # MI vê uma dependência forte e real


@pytest.mark.unit
def test_mi_ranks_correlated_hvi_features_above_uncorrelated_ones():
    """Aplicação real: nos dados HVI do Módulo 0, MI deveria ser bem
    maior entre features correlacionadas por construção (uhml,
    uniformity — correlação 0.55) do que entre features independentes
    por construção (micronaire, rd — correlação 0)."""
    spec = default_spec()
    bales = generate_bales(spec, n=3000, seed=2024)
    idx = {name: i for i, name in enumerate(spec.features)}

    mi_correlated = mutual_information_binned(
        bales[:, idx["uhml"]], bales[:, idx["uniformity"]]
    )
    mi_uncorrelated = mutual_information_binned(
        bales[:, idx["micronaire"]], bales[:, idx["rd"]]
    )

    assert mi_correlated > mi_uncorrelated * 3
