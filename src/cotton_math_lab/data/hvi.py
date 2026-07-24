"""Gerador sintético de dados HVI de fardos de algodão.

Os parâmetros populacionais são *conhecidos por construção*: isso permite que
módulos posteriores (MLE, PCA, informação mútua) sejam validados contra a
verdade que plantamos aqui, e não contra uma referência externa."""

from dataclasses import dataclass

import numpy as np

FEATURES: tuple[str, ...] = (
    "micronaire",   # índice de finura/maturidade
    "uhml",         # comprimento (mm)
    "uniformity",   # índice de uniformidade (%)
    "strength",     # resistência (g/tex)
    "elongation",   # elongação (%)
    "rd",           # refletância (grau de brancura)
    "plus_b",       # amarelamento
    "trash",        # área de impurezas (%)
)

@dataclass(frozen=True, eq=False)
class HVISpec:
    """Parâmetros populacionais de uma distribuição normal multivariada de HVI."""

    features: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    correlation: np.ndarray

    def __post_init__(self) -> None:
        # arrays imutáveis: o spec é a "verdade" do experimento e não deve
        # ser alterado por engano em nenhum ponto do laboratório
        for arr in (self.means, self.stds, self.correlation):
            arr.setflags(write=False)

    @property
    def covariance(self) -> np.ndarray:
        """Σ = D · R · D, onde D = diag(σ)."""
        d = np.diag(self.stds)
        return d @ self.correlation @ d


def _build_correlation(pairs: dict[tuple[str, str], float]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(FEATURES)}
    corr = np.eye(len(FEATURES), dtype=np.float64)
    for (a, b), value in pairs.items():
        corr[idx[a], idx[b]] = value
        corr[idx[b], idx[a]] = value
    return corr

def default_spec() -> HVISpec:
    """Spec com valores típicos de algodão upland brasileiro."""
    means = np.array([4.30, 29.0, 82.5, 30.0, 6.5, 75.0, 8.5, 0.60])
    stds = np.array([0.40, 1.20, 1.50, 2.50, 0.80, 3.00, 1.00, 0.30])

    correlation = _build_correlation(
        {
            # maturidade da fibra puxa micronaire e resistência juntos
            ("micronaire", "strength"): 0.25,
            ("micronaire", "elongation"): 0.20,
            # fibra mais longa tende a ser mais fina
            ("micronaire", "uhml"): -0.15,
            # comprimento, uniformidade e resistência formam o bloco "fibra"
            ("uhml", "uniformity"): 0.55,
            ("uhml", "strength"): 0.45,
            ("uniformity", "strength"): 0.30,
            # trade-off clássico: resistência × elongação
            ("strength", "elongation"): -0.20,
            # bloco "cor": mais branco, menos amarelo
            ("rd", "plus_b"): -0.45,
            # impureza suja a cor
            ("trash", "rd"): -0.35,
            ("trash", "plus_b"): 0.20,
        }
    )

    return HVISpec(
        features=FEATURES,
        means=means,
        stds=stds,
        correlation=correlation,
    )


def generate_bales(spec: HVISpec, n: int, seed: int) -> np.ndarray:
    """Amostra `n` fardos da normal multivariada definida por `spec`.

    Usa a decomposição de Cholesky Σ = L·Lᵀ: se z ~ N(0, I), então
    μ + L·z ~ N(μ, Σ). Retorna um array (n, k) de float64.
    """
    rng = np.random.default_rng(seed)
    lower = np.linalg.cholesky(spec.covariance)
    standard = rng.standard_normal((n, len(spec.features)))
    return spec.means + standard @ lower.T
