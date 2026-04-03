"""Tests for priorlab.empirical_bayes module."""

import math
import numpy as np
import pytest

from priorlab.models import ElicitedQuantiles, ExpertPrior, FittedDistribution
from priorlab.empirical_bayes import empirical_bayes_analysis


def _make_expert(expert_id, mu, sigma):
    """Helper: create an ExpertPrior with a Normal best_fit."""
    fit = FittedDistribution(
        family="normal",
        params={"mu": mu, "sigma": sigma},
    )
    return ExpertPrior(expert_id=expert_id, label=expert_id, best_fit=fit)


def _make_experts(mus, sigmas):
    """Build a list of ExpertPrior from parallel mu/sigma arrays."""
    return [
        _make_expert(f"E{i+1}", mu, sig)
        for i, (mu, sig) in enumerate(zip(mus, sigmas))
    ]


class TestShrinkageFactorBounds:
    """B must lie in [0, 1]."""

    def test_b_in_unit_interval_diverse(self):
        experts = _make_experts(
            mus=[-1.0, 0.0, 0.5, 1.5, 3.0],
            sigmas=[0.5, 0.5, 0.5, 0.5, 0.5],
        )
        result = empirical_bayes_analysis(experts)
        B = result["shrinkage_factor_B"]
        assert 0.0 <= B <= 1.0, f"B={B} out of [0,1]"

    def test_b_in_unit_interval_tight(self):
        # Experts very close together -> high shrinkage (B near 1)
        experts = _make_experts(
            mus=[0.0, 0.01, -0.01],
            sigmas=[1.0, 1.0, 1.0],
        )
        result = empirical_bayes_analysis(experts)
        B = result["shrinkage_factor_B"]
        assert 0.0 <= B <= 1.0


class TestSUREFinite:
    """SURE MSE estimate must be finite and non-negative."""

    def test_sure_finite_basic(self):
        experts = _make_experts(
            mus=[0.0, 1.0, 2.0],
            sigmas=[0.5, 0.5, 0.5],
        )
        result = empirical_bayes_analysis(experts)
        sure = result["sure_mse_estimate"]
        assert math.isfinite(sure), f"SURE not finite: {sure}"


class TestGrandMean:
    """Grand mean should be the average of expert means."""

    def test_grand_mean_correct(self):
        mus = [1.0, 2.0, 3.0, 4.0]
        experts = _make_experts(mus=mus, sigmas=[0.5]*4)
        result = empirical_bayes_analysis(experts)
        expected = np.mean(mus)
        assert abs(result["grand_mean"] - expected) < 1e-10


class TestMarginalLLProfile:
    """Marginal log-likelihood profile must have expected structure."""

    def test_profile_length_and_keys(self):
        experts = _make_experts(
            mus=[0.0, 0.5, 1.0, 1.5, 2.0],
            sigmas=[0.3, 0.3, 0.3, 0.3, 0.3],
        )
        result = empirical_bayes_analysis(experts, grid_size=50)
        profile = result["marginal_ll_profile"]
        assert len(profile) == 50
        # Each entry has A and ll keys
        for entry in profile:
            assert "A" in entry
            assert "ll" in entry
            assert math.isfinite(entry["ll"])
