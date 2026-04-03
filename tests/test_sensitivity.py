"""Tests for prior sensitivity analysis."""

import numpy as np
import pytest

from priorlab.sensitivity import (
    local_sensitivity,
    global_sensitivity,
    expert_influence,
    sensitivity_analysis,
)


def test_shrinkage_factor_in_unit_interval():
    """d(mu_post)/d(mu0) = sigma_y^2/(sigma0^2+sigma_y^2) must be in (0,1)."""
    for sigma0 in [0.1, 0.5, 1.0, 5.0, 100.0]:
        for sigma_y in [0.1, 0.5, 1.0, 5.0]:
            result = local_sensitivity(mu0=0.0, sigma0=sigma0, y=1.0, sigma_y=sigma_y)
            sf = result["d_mu_post_d_mu0"]
            assert 0 < sf < 1, f"Shrinkage {sf} out of (0,1) for sigma0={sigma0}, sigma_y={sigma_y}"


def test_shrinkage_increases_with_vague_prior():
    """Vaguer prior (larger sigma0) => shrinkage closer to 0 (data dominates)."""
    narrow = local_sensitivity(mu0=0.0, sigma0=0.1, y=1.0, sigma_y=1.0)
    wide = local_sensitivity(mu0=0.0, sigma0=10.0, y=1.0, sigma_y=1.0)
    # Wider prior => smaller d_mu_post/d_mu0 (less influence of prior)
    assert wide["d_mu_post_d_mu0"] < narrow["d_mu_post_d_mu0"]


def test_global_sweep_monotone_mu():
    """Posterior mean should increase monotonically as mu0 increases."""
    result = global_sensitivity(mu0=0.0, sigma0=1.0, y=1.0, sigma_y=1.0, n_steps=20)
    means = [pt["post_mean"] for pt in result["mu_sweep"]]
    for i in range(1, len(means)):
        assert means[i] >= means[i - 1] - 1e-12, "Posterior mean should be non-decreasing in mu0"


def test_robustness_ratio_near_one_for_tight_data():
    """With very precise data (small sigma_y), prior barely matters => ratio ~ 1."""
    result = global_sensitivity(mu0=0.0, sigma0=1.0, y=1.0, sigma_y=0.01, n_steps=20)
    assert result["robustness_ratio"] < 1.05, f"Ratio {result['robustness_ratio']} should be near 1"


def test_expert_influence_sums_near_zero():
    """Leave-one-out influences should roughly balance (net close to 0 for equal priors)."""
    experts = [
        {"mu0": -0.5, "sigma0": 1.0},
        {"mu0": 0.0, "sigma0": 1.0},
        {"mu0": 0.5, "sigma0": 1.0},
    ]
    infl = expert_influence(experts, y=0.0, sigma_y=1.0)
    assert len(infl) == 3
    # Removing symmetric extremes should give opposite signs
    assert infl[0] * infl[2] < 0 or abs(infl[0]) < 1e-6, (
        "Removing low vs high expert should shift posterior oppositely"
    )
