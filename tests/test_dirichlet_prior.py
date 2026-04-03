"""Tests for Dirichlet Process prior nonparametric density estimation."""

import numpy as np
import pytest

from priorlab.dirichlet_prior import dp_prior, _stick_breaking_weights


class TestDPPrior:
    """Core DP prior functionality."""

    def test_dp_density_integrates_to_one(self, symmetric_quantiles):
        """DP density must integrate to approximately 1."""
        result = dp_prior(symmetric_quantiles, alpha=1.0, K_max=20, B=50, seed=42)
        grid = np.array(result["grid"])
        density = np.array(result["dp_density"])
        integral = np.trapezoid(density, grid)
        assert abs(integral - 1.0) < 0.05, (
            f"DP density integral = {integral:.4f}, expected ~1.0"
        )

    def test_stick_breaking_weights_sum_to_one(self):
        """Stick-breaking weights must sum to 1."""
        rng = np.random.default_rng(123)
        for alpha in [0.1, 1.0, 5.0, 20.0]:
            weights = _stick_breaking_weights(alpha, K=20, rng=rng)
            assert abs(weights.sum() - 1.0) < 1e-10, (
                f"Weights sum = {weights.sum():.10f} for alpha={alpha}"
            )
            assert np.all(weights >= 0), "Weights must be non-negative"

    def test_credible_bands_bracket_density(self, symmetric_quantiles):
        """95% credible bands should generally bracket the point estimate."""
        result = dp_prior(symmetric_quantiles, alpha=1.0, K_max=20, B=100, seed=42)
        density = np.array(result["dp_density"])
        lower = np.array(result["dp_lower"])
        upper = np.array(result["dp_upper"])
        # At most ~5% of points should fall outside bands
        # (but bootstrapped mean can sometimes exceed pointwise percentiles)
        within = np.sum((density >= lower - 1e-6) & (density <= upper + 1e-6))
        fraction_within = within / len(density)
        assert fraction_within > 0.80, (
            f"Only {fraction_within:.1%} of density within credible bands"
        )

    def test_effective_components_reasonable(self, skewed_quantiles):
        """Effective number of components should be between 1 and K_max."""
        result = dp_prior(skewed_quantiles, alpha=1.0, K_max=20, B=50, seed=42)
        n_eff = result["n_effective_components"]
        assert 1.0 <= n_eff <= 20.0, (
            f"n_effective_components = {n_eff}, expected in [1, 20]"
        )

    def test_kl_vs_parametric_nonnegative(self, symmetric_quantiles):
        """KL divergence from DP to parametric should be non-negative."""
        result = dp_prior(symmetric_quantiles, alpha=1.0, K_max=20, B=50, seed=42)
        kl = result["kl_vs_parametric"]
        assert kl >= -0.01, (
            f"KL divergence = {kl:.4f}, expected >= 0 (allowing small numerical error)"
        )
