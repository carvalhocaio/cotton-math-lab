"""JS como detector de drift — simétrica e limitada, ao contrário de KL."""

import numpy as np

from cotton_math_lab.infotheory.discrete import kl_divergence_discrete


def js_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    """D_JS(p,q) = ½·D_KL(p‖m) + ½·D_KL(q‖m), com m = (p+q)/2.

    Ao contrário de KL, é simétrica e limitada (0 ≤ D_JS ≤ log 2) —
    propriedades que fazem dela uma escolha melhor que KL pra medir
    "quão diferentes" duas distribuições são, mesmo não sendo uma métrica
    formal (a raiz quadrada, não D_JS em si, satisfaz desigualdade
    triangular — é a distância de Jensen-Shannon).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_discrete(p, m) + 0.5 * kl_divergence_discrete(q, m)


def histogram_distribution(
    values: np.ndarray, bin_edges: np.ndarray, smoothing: float = 1e-6
) -> np.ndarray:
    """Amostra contínua -> distribuição discreta via histograma, com
    suavização mínima pra evitar probabilidade zero absoluta (que
    quebraria KL/JS por log(0))."""
    counts, _ = np.histogram(values, bins=bin_edges)
    counts = counts.astype(np.float64) + smoothing
    return counts / counts.sum()


def detect_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 20,
    threshold: float = 0.05,
) -> tuple[bool, float]:
    """Compara a distribuição de `current` contra `baseline` via JS.

    Retorna (houve_drift, divergência_js). O threshold é uma escolha de
    projeto, não um valor universal — calibrável a partir de dado
    histórico de quanto a variação natural entre safras genuinamente
    semelhantes costuma produzir.
    """
    combined = np.concatenate([baseline, current])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    p = histogram_distribution(baseline, bin_edges)
    q = histogram_distribution(current, bin_edges)

    divergence = js_divergence_discrete(p, q)
    return divergence > threshold, divergence
