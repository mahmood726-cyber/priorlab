"""Tests for robust Bayesian analysis module."""

import numpy as np
import pytest

from priorlab.robust_bayes import (
    epsilon_contamination,
    gamma_minimax,
    prior_range_analysis,
    density_bands,
    robust_bayesian_analysis,
)


class TestRobustBayes:
    """Robust Bayesian analysis tests."""

    def test_robustness_index_in_zero_one(self, symmetric_quantiles):
        """Robustness index must be in [0, 0.5] (max eps tested)."""
        result = gamma_minimax(symmetric_quantiles, y=-0.2, sigma_y=0.1)
        ri = result["robustness_index"]
        assert 0.0 <= ri <= 0.5, (
            f"Robustness index = {ri}, expected in [0, 0.5]"
        )
        assert result["gamma_minimax_decision"] in ("treat", "do_not_treat")

    def test_eps_zero_gives_point_estimate(self, symmetric_quantiles):
        """At eps=0, lower and upper posterior mean should be equal."""
        ranges = epsilon_contamination(
            symmetric_quantiles, y=-0.2, sigma_y=0.1,
            eps_values=[0.0]
        )
        r0 = ranges[0.0]
        assert abs(r0["lower"] - r0["upper"]) < 1e-10, (
            f"At eps=0: lower={r0['lower']}, upper={r0['upper']} should be equal"
        )

    def test_range_widens_with_eps(self, symmetric_quantiles):
        """Posterior mean range must widen as eps increases."""
        eps_vals = [0.0, 0.1, 0.3, 0.5]
        ranges = epsilon_contamination(
            symmetric_quantiles, y=-0.2, sigma_y=0.1,
            eps_values=eps_vals
        )
        widths = [ranges[e]["upper"] - ranges[e]["lower"] for e in eps_vals]
        for i in range(1, len(widths)):
            assert widths[i] >= widths[i - 1] - 1e-10, (
                f"Width at eps={eps_vals[i]} ({widths[i]:.4f}) < "
                f"width at eps={eps_vals[i-1]} ({widths[i-1]:.4f})"
            )

    def test_prior_range_four_corners(self):
        """Prior range analysis should return 4 corner results."""
        results = prior_range_analysis(
            y=0.5, sigma_y=0.2,
            mu_range=(0.0, 1.0),
            sigma_range=(0.1, 0.5)
        )
        assert len(results) == 4, f"Expected 4 corners, got {len(results)}"
        for r in results:
            assert "post_mean" in r
            assert "post_ci" in r
            assert len(r["post_ci"]) == 2
            assert r["post_ci"][0] < r["post_ci"][1], "CI lower must < upper"

    def test_full_analysis_structure(self, symmetric_quantiles):
        """Full analysis returns all expected keys with correct types."""
        result = robust_bayesian_analysis(
            symmetric_quantiles, y=-0.2, sigma_y=0.1
        )
        assert "posterior_mean_range_by_eps" in result
        assert "robustness_index" in result
        assert "gamma_minimax_decision" in result
        assert "prior_range_results" in result
        assert "density_upper" in result
        assert "density_lower" in result
        assert "is_robust" in result
        assert isinstance(result["is_robust"], bool)
        assert isinstance(result["robustness_index"], float)
        assert 0.0 <= result["robustness_index"] <= 0.5
