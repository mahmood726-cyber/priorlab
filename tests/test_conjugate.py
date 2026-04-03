"""Tests for conjugate prior analysis."""

import numpy as np
import pytest

from priorlab.conjugate import (
    normal_normal,
    beta_binomial,
    gamma_poisson,
    conjugate_analysis,
)


def test_posterior_precision_exceeds_prior():
    """After observing data, posterior precision must exceed prior precision."""
    result = normal_normal(mu0=0.0, sigma0=1.0, y=1.0, sigma_y=1.0)
    assert result["posterior_precision"] > result["prior_precision"]


def test_beta_binomial_posterior_concentration():
    """After 10 successes in 10 trials, posterior mean > prior mean."""
    result = beta_binomial(a=1.0, b=1.0, x=10, n=10)
    assert result["posterior_mean"] > result["prior_mean"]
    assert result["posterior_precision"] > result["prior_precision"]


def test_gamma_poisson_posterior_update():
    """After observing counts, posterior rate increases."""
    result = gamma_poisson(a=2.0, b=1.0, sum_x=5, n=3)
    assert result["b_posterior"] == 1.0 + 3  # b + n
    assert result["a_posterior"] == 2.0 + 5  # a + sum_x
    assert result["posterior_precision"] > result["prior_precision"]


def test_conjugate_analysis_returns_all_pairs(symmetric_quantiles):
    """Full analysis must return all four conjugate pairs."""
    result = conjugate_analysis(eq=symmetric_quantiles, y=0.5, n=5, sigma_y=1.0)
    for key in ["normal_normal", "normal_ig", "beta_binomial", "gamma_poisson"]:
        assert key in result, f"Missing conjugate pair: {key}"


def test_shrinkage_between_zero_and_one():
    """Normal-Normal shrinkage factor must be in (0, 1)."""
    for sigma0 in [0.1, 0.5, 1.0, 5.0]:
        for sigma_y in [0.1, 0.5, 1.0, 5.0]:
            result = normal_normal(mu0=0.0, sigma0=sigma0, y=1.0, sigma_y=sigma_y)
            assert 0 < result["shrinkage"] < 1, (
                f"Shrinkage {result['shrinkage']} out of (0,1)"
            )
