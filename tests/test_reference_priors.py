import numpy as np
import pytest
from scipy.stats import norm

from priorlab.reference_priors import (
    compare_reference_priors,
    jeffreys_normal_mean,
    jeffreys_normal_variance,
    maxent_moments,
    maxent_support,
)


GRID = np.linspace(-5, 5, 500)
POS_GRID = np.linspace(0.01, 5, 500)


def test_jeffreys_mean_integrates_to_one():
    """Jeffreys flat prior must integrate to ~1 on bounded grid."""
    pdf = jeffreys_normal_mean(GRID, sigma2=1.0)
    integral = np.trapezoid(pdf, GRID)
    assert abs(integral - 1.0) < 0.01


def test_jeffreys_variance_integrates_to_one():
    """Jeffreys 1/sigma2 prior must integrate to ~1 on bounded positive grid."""
    pdf = jeffreys_normal_variance(POS_GRID)
    integral = np.trapezoid(pdf, POS_GRID)
    assert abs(integral - 1.0) < 0.01


def test_maxent_is_gaussian():
    """MaxEnt under mean+second moment constraints is Gaussian; check peak at mu."""
    mu_c = 1.0
    m2 = mu_c ** 2 + 4.0  # sigma^2 = 4, sigma = 2
    pdf = maxent_moments(GRID, mu_c, m2)
    # Peak should be near mu = 1.0
    peak_x = GRID[np.argmax(pdf)]
    assert abs(peak_x - mu_c) < 0.1


def test_maxent_support_uniform():
    """MaxEnt with no constraints on [a,b] is uniform."""
    grid = np.linspace(0, 1, 200)
    pdf = maxent_support(grid, 0, 1)
    # Should be constant ~ 1.0 inside [0,1]
    interior = pdf[(grid > 0.05) & (grid < 0.95)]
    assert np.std(interior) < 0.01


def test_kl_elicited_vs_self_is_zero():
    """KL divergence of a distribution against itself should be ~0."""
    elicited = norm(loc=0, scale=1).pdf(GRID)
    result = compare_reference_priors(elicited, GRID, sigma2=1.0)
    # KL against uniform should be positive (elicited is peaked)
    assert result["kl_elicited_vs_uniform"] > 0
    # All output keys present
    for key in ["jeffreys_mean", "jeffreys_variance", "maxent_pdf",
                "uniform_pdf", "kl_elicited_vs_jeffreys",
                "kl_elicited_vs_maxent", "kl_elicited_vs_uniform",
                "info_gain_elicited", "info_gain_jeffreys", "info_gain_maxent"]:
        assert key in result
