"""Beta-Binomial conjugado: proporção de fardos fora de especificação."""


def beta_binomial_posterior(
    k: int, n: int, alpha_prior: float, beta_prior: float
) -> tuple[float, float]:
    """Parâmetros da posterior, dado prior p ~ Beta(α, β) e k sucessos em
    n ensaios ~ Binomial(n, p).

    Conjugação exata: a posterior é Beta(α+k, β+n-k) — a mesma família do
    prior, só com os parâmetros deslocados pelos dados observados. Não há
    aproximação nenhuma aqui, ao contrário do Normal-Normal do ciclo
    anterior (que também é exato, mas exige σ conhecido); Beta-Binomial é
    exato incondicionalmente.
    """
    return alpha_prior + k, beta_prior + (n - k)


def posterior_mean(alpha: float, beta: float) -> float:
    """Média da distribuição Beta(α, β) — a estimativa pontual bayesiana."""
    return alpha / (alpha + beta)
