"""IC (frequentista) vs. intervalo de credibilidade (bayesiano) - a mesma
pergunta aparente, duas garantias matemáticas diferentes."""

import numpy as np
from scipy import stats


def wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """IC de Wilson para uma proporção binomial.

    Não usa nenhum prior — é a inversão de um teste de hipótese: o
    conjunto de valores p₀ para os quais um teste de score (H₀: p=p₀) não
    seria rejeitado no nível de significância dado. A garantia
    frequentista vem inteiramente dessa construção, e não depende de
    nenhuma crença prévia sobre p.
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return center - margin, center + margin


def beta_credible_interval(
    alpha: float, beta: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Intervalo de credibilidade a partir dos quantis de uma posterior Beta.

    Diferente do IC: aqui p É tratado como variável aleatória (dada a
    posterior), e o intervalo é literalmente "95% de probabilidade
    posterior de p estar aqui" — uma afirmação direta sobre o parâmetro,
    condicionada nos dados observados E no prior escolhido. Essa condição
    extra ("E no prior") é o que falta na leitura popular incorreta do IC.
    """
    tail = (1 - confidence) / 2
    return stats.beta.ppf(tail, alpha, beta), stats.beta.ppf(1 - tail, alpha, beta)
