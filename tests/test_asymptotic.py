"""Tests for asymptotic theory of prior impact."""

import numpy as np
import pytest

from priorlab.asymptotic import (
    bvm_convergence,
    influence_decay,
    effective_prior_n,
    concentration_rate,
    asymptotic_analysis,
)


def test_bvm_tv_decreases_with_n():
    """Total variation distance to BvM Normal must decrease as n grows."""
    results = bvm_convergence(mu0=2.0, sigma0=1.0, y=0.0, sigma_y=1.0,
                              ns=[1, 5, 10, 50, 100, 500])
    tvs = [r["tv_distance"] for r in results]
    # Each should be <= the previous (monotone decrease)
    for i in range(1, len(tvs)):
        assert tvs[i] <= tvs[i - 1] + 1e-6, (
            f"TV not decreasing: n={results[i]['n']} tv={tvs[i]} > prev {tvs[i-1]}"
        )


def test_concentration_alpha_near_half():
    """For Normal posterior, concentration rate alpha should be near 0.5."""
    result = concentration_rate(mu0=0.0, sigma0=1.0, y=1.0, sigma_y=1.0, n_max=200)
    alpha = result["concentration_alpha"]
    assert 0.4 < alpha < 0.6, f"Alpha = {alpha}, expected near 0.5"


def test_effective_prior_n_positive():
    """Effective prior sample size must be > 0."""
    for sigma0 in [0.1, 0.5, 1.0, 5.0]:
        for sigma_y in [0.1, 0.5, 1.0, 5.0]:
            n0 = effective_prior_n(sigma0, sigma_y)
            assert n0 > 0, f"Effective n = {n0} <= 0"


def test_influence_decay_decreasing():
    """Influence ratio should generally decrease with n."""
    results = influence_decay(mu0=3.0, sigma0=1.0, y=0.0, sigma_y=1.0, n_max=50)
    # First ratio should be > last ratio
    assert results[0]["ratio"] > results[-1]["ratio"], "Influence should decay"


def test_asymptotic_analysis_keys():
    """Full analysis must return all expected keys."""
    result = asymptotic_analysis(y=1.0, sigma_y=1.0, experts=[
        {"expert_id": "E1", "mu0": -1.0, "sigma0": 1.0},
        {"expert_id": "E2", "mu0": 1.0, "sigma0": 1.0},
    ])
    for key in ["bvm_convergence", "influence_decay", "effective_prior_n",
                "concentration_rate_alpha", "concentration_c", "robustness_horizons"]:
        assert key in result, f"Missing key: {key}"
    assert result["effective_prior_n"] > 0
    assert len(result["robustness_horizons"]) == 2
