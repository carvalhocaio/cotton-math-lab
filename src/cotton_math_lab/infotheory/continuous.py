"""Continuous KL via numerical grid integration, and the demonstration
of the forward vs. reverse KL asymmetry."""

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def kl_divergence_grid(
    p_vals: np.ndarray, q_vals: np.ndarray, x_grid: np.ndarray
) -> float:
    """D_KL(p‖q) for continuous densities, via the trapezoidal rule on a
    shared grid."""
    p_vals = np.clip(p_vals, 1e-300, None)
    q_vals = np.clip(q_vals, 1e-300, None)
    integrand = p_vals * (np.log(p_vals) - np.log(q_vals))
    return float(np.trapezoid(integrand, x_grid))


def fit_gaussian_by_forward_kl(target_pdf, x_grid: np.ndarray) -> tuple[float, float]:
    """Minimizes D_KL(p‖q) over a Gaussian q. Has a closed form: moment
    matching — μ = E_p[x], σ² = Var_p[x]. This is KL's moment-projection
    property for exponential families (Gaussian is one of them):
    minimizing forward KL over an exponential family always reduces to
    matching the sufficient statistics, regardless of p's shape.
    """
    p_vals = target_pdf(x_grid)
    p_vals = p_vals / np.trapezoid(p_vals, x_grid)
    mean = np.trapezoid(x_grid * p_vals, x_grid)
    variance = np.trapezoid((x_grid - mean) ** 2 * p_vals, x_grid)
    return float(mean), float(np.sqrt(variance))


def fit_gaussian_by_reverse_kl(
    target_pdf, x_grid: np.ndarray, init_mu: float, init_sigma: float = 1.0
) -> tuple[float, float]:
    """Minimizes D_KL(q‖p) over a Gaussian q. No general closed form —
    numerical optimization, and the result depends on the starting point
    when p is multimodal: the reverse KL surface has a local minimum
    near each mode of p, because q is penalized for placing mass where p
    is low, but is never required to cover all of p's mass.
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
