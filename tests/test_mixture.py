"""Tests for mixture prior construction via EM algorithm."""

import numpy as np
import pytest
from scipy import stats

from priorlab.mixture import fit_mixture, _em_fit


def test_em_converges_on_unimodal():
    """EM should converge for a simple unimodal Normal density."""
    grid = np.linspace(-4, 4, 200)
    density = stats.norm.pdf(grid, loc=0, scale=1)
    result = _em_fit(grid, density, k=1, max_iter=100, tol=1e-6, seed=42)
    assert result["converged"] is True
    assert abs(result["means"][0] - 0.0) < 0.2
    assert abs(result["variances"][0] - 1.0) < 0.5


def test_bic_selects_one_component_for_unimodal():
    """BIC should select K=1 for a clean unimodal Normal density."""
    grid = np.linspace(-5, 5, 300)
    density = stats.norm.pdf(grid, loc=0, scale=1)
    result = fit_mixture(grid, density, max_k=3, seed=42)
    assert result["n_components"] == 1
    assert 1 in result["bic_scores"]
    assert 2 in result["bic_scores"]
    assert 3 in result["bic_scores"]


def test_mixture_pdf_integrates_to_one():
    """The fitted mixture PDF should integrate to approximately 1."""
    grid = np.linspace(-5, 5, 300)
    density = stats.norm.pdf(grid, loc=0, scale=1)
    result = fit_mixture(grid, density, max_k=2, seed=42)
    integral = np.trapezoid(result["mixture_pdf"], grid)
    assert abs(integral - 1.0) < 0.05, f"Mixture integral {integral} should be ~1"


def test_weights_sum_to_one():
    """Component weights must sum to 1."""
    grid = np.linspace(-6, 6, 300)
    density = 0.5 * stats.norm.pdf(grid, loc=-2, scale=0.8) + \
              0.5 * stats.norm.pdf(grid, loc=2, scale=0.8)
    result = fit_mixture(grid, density, max_k=3, seed=42)
    assert abs(sum(result["weights"]) - 1.0) < 1e-6


def test_bimodal_detects_two_components():
    """A clearly bimodal density should yield K >= 2 components."""
    grid = np.linspace(-8, 8, 400)
    density = 0.5 * stats.norm.pdf(grid, loc=-3, scale=0.5) + \
              0.5 * stats.norm.pdf(grid, loc=3, scale=0.5)
    result = fit_mixture(grid, density, max_k=3, seed=42)
    assert result["n_components"] >= 2, (
        f"Expected >=2 components for bimodal, got {result['n_components']}"
    )
    # Means should be near -3 and 3
    means = sorted(result["means"])
    if result["n_components"] >= 2:
        assert means[0] < 0, f"First mean {means[0]} should be negative"
        assert means[-1] > 0, f"Last mean {means[-1]} should be positive"
