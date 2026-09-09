import numpy as np
import pytest
from scipy import stats

from cotton_math_lab.infotheory.continuous import (
    fit_gaussian_by_forward_kl,
    fit_gaussian_by_reverse_kl,
    kl_divergence_grid,
)

X_GRID = np.linspace(-10, 10, 4000)


def _bimodal_target(x):
    return 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)


@pytest.mark.oracle
def test_kl_grid_matches_closed_form_gaussian_kl():
    """Validates the numerical integration against the closed-form KL
    between two Gaussians — no approximation assumed, only checked."""
    mu1, sigma1 = -1.0, 1.5
    mu2, sigma2 = 2.0, 0.8
    fine_grid = np.linspace(-20, 20, 200_000)

    p_vals = stats.norm.pdf(fine_grid, mu1, sigma1)
    q_vals = stats.norm.pdf(fine_grid, mu2, sigma2)
    grid_result = kl_divergence_grid(p_vals, q_vals, fine_grid)

    closed_form = (
        np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5
    )
    assert grid_result == pytest.approx(closed_form, abs=1e-6)


@pytest.mark.unit
def test_forward_kl_spreads_to_cover_both_modes():
    """Forward KL (mean-seeking): the optimal Gaussian covers both modes
    — mean exactly in the middle, standard deviation large enough that
    neither mode is left with too-low density."""
    mu, sigma = fit_gaussian_by_forward_kl(_bimodal_target, X_GRID)

    assert mu == pytest.approx(0.0, abs=1e-6)
    assert sigma > 2.5  # much wider than either component (sigma=1)


@pytest.mark.unit
def test_reverse_kl_collapses_onto_one_mode():
    """Reverse KL (mode-seeking): starting near the left mode, the
    optimal Gaussian collapses onto it — mean and standard deviation
    practically replicate ONE component of the mixture, ignoring the
    other."""
    mu, sigma = fit_gaussian_by_reverse_kl(_bimodal_target, X_GRID, init_mu=-2.5)

    assert mu == pytest.approx(-3.0, abs=0.1)
    assert sigma == pytest.approx(1.0, abs=0.1)


@pytest.mark.unit
def test_reverse_kl_result_depends_on_initialization():
    """Unlike forward KL (closed form, unique optimum), reverse KL has
    multiple local optima — the result depends on where the
    optimization starts, the signature of a non-convex surface."""
    mu_left, _ = fit_gaussian_by_reverse_kl(_bimodal_target, X_GRID, init_mu=-2.5)
    mu_right, _ = fit_gaussian_by_reverse_kl(_bimodal_target, X_GRID, init_mu=2.5)

    assert mu_left == pytest.approx(-3.0, abs=0.1)
    assert mu_right == pytest.approx(3.0, abs=0.1)
    assert abs(mu_left - mu_right) > 5.0


@pytest.mark.unit
def test_forward_kl_fit_is_wider_than_reverse_kl_fit():
    """The numerical signature of the asymmetry, in a single number:
    forward KL produces a much larger sigma than reverse KL, on the
    same bimodal target."""
    _, sigma_forward = fit_gaussian_by_forward_kl(_bimodal_target, X_GRID)
    _, sigma_reverse = fit_gaussian_by_reverse_kl(_bimodal_target, X_GRID, init_mu=-2.5)

    assert sigma_forward > sigma_reverse * 2
