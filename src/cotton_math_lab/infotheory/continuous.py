"""KL contínua via integração numérica em grade, e a demonstração da
assimetria forward vs. reverse KL."""

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def kl_divergence_grid(
    p_vals: np.ndarray, q_vals: np.ndarray, x_grid: np.ndarray
) -> float:
    """D_KL(p‖q) para densidades contínuas, via regra do trapézio numa
    grade compartilhada."""
    p_vals = np.clip(p_vals, 1e-300, None)
    q_vals = np.clip(q_vals, 1e-300, None)
    integrand = p_vals * (np.log(p_vals) - np.log(q_vals))
    return float(np.trapezoid(integrand, x_grid))


def fit_gaussian_by_forward_kl(target_pdf, x_grid: np.ndarray) -> tuple[float, float]:
    """Minimiza D_KL(p‖q) sobre q Gaussiana. Tem forma fechada: casamento
    de momentos — μ = E_p[x], σ² = Var_p[x]. É a propriedade de projeção
    de momento da KL para famílias exponenciais (Gaussiana é uma delas):
    minimizar forward KL sobre uma família exponencial sempre reduz a
    igualar os momentos suficientes, não importa a forma de p.
    """
    p_vals = target_pdf(x_grid)
    p_vals = p_vals / np.trapezoid(p_vals, x_grid)
    mean = np.trapezoid(x_grid * p_vals, x_grid)
    variance = np.trapezoid((x_grid - mean) ** 2 * p_vals, x_grid)
    return float(mean), float(np.sqrt(variance))


def fit_gaussian_by_reverse_kl(
    target_pdf, x_grid: np.ndarray, init_mu: float, init_sigma: float = 1.0
) -> tuple[float, float]:
    """Minimiza D_KL(q‖p) sobre q Gaussiana. Sem forma fechada geral —
    otimização numérica, e o resultado depende do ponto de partida quando
    p é multimodal: a superfície de reverse KL tem um mínimo local perto
    de cada modo de p, porque q é penalizada por colocar massa onde p é
    baixo, mas não é obrigada a cobrir toda a massa de p.
    """
    p_vals = target_pdf(x_grid)
    p_vals = p_vals / np.trapezoid(p_vals, x_grid)

    def objective(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        q_vals = stats.norm.pdf(x_grid, mu, sigma)
        return kl_divergence_grid(q_vals, p_vals, x_grid)

    result = minimize(objective, x0=[init_mu, np.log(init_sigma)], method="Nelder-Mead")
    mu_opt, log_sigma_opt = result.x
    return float(mu_opt), float(np.exp(log_sigma_opt))
