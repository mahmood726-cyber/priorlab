import numpy as np
import pytest

from priorlab.pc_priors import (
    compute_lambda,
    compute_pc_priors,
    halfcauchy_density,
    pc_prior_tau,
    pc_prior_tau2,
)


def test_lambda_calculation():
    """lambda = -log(alpha)/U.  P(tau>1) = 0.05 => lambda = -log(0.05)/1."""
    lam = compute_lambda(upper_bound=1.0, alpha=0.05)
    expected = -np.log(0.05)  # ~2.996
    assert abs(lam - expected) < 1e-10


def test_pc_tau_integrates_to_one():
    """Exponential density integrates to 1 over [0, inf) — check on large grid."""
    lam = compute_lambda(1.0, 0.05)
    grid = np.linspace(1e-6, 20, 2000)
    density = pc_prior_tau(grid, lam)
    integral = np.trapezoid(density, grid)
    assert abs(integral - 1.0) < 0.01


def test_pc_tau2_positive_and_finite():
    """PC prior on variance scale should be positive for tau^2 > 0."""
    lam = compute_lambda(0.5, 0.10)
    tau2_grid = np.linspace(0.001, 2.0, 500)
    density = pc_prior_tau2(tau2_grid, lam)
    assert np.all(density > 0)
    assert np.all(np.isfinite(density))


def test_compute_pc_priors_returns_all_keys():
    """Full computation should return all expected keys."""
    result = compute_pc_priors(upper_bound=1.0, alpha=0.05, n_grid=300)
    expected_keys = {
        "tau_grid", "pc_tau_density", "pc_tau2_density",
        "halfcauchy_density", "lambda_param", "kl_pc_vs_halfcauchy",
    }
    assert set(result.keys()) == expected_keys
    assert len(result["tau_grid"]) == 300
    assert result["lambda_param"] > 0
    # KL divergence should be non-negative
    assert result["kl_pc_vs_halfcauchy"] >= 0


def test_halfcauchy_integrates_to_one():
    """Half-Cauchy density integrates to 1 over [0, inf)."""
    grid = np.linspace(1e-6, 100, 5000)
    density = halfcauchy_density(grid, scale=1.0)
    integral = np.trapezoid(density, grid)
    assert abs(integral - 1.0) < 0.01
