"""Tests for priorlab.laplace_approximation module."""

import math
import numpy as np
import pytest

from priorlab.models import ElicitedQuantiles
from priorlab.laplace_approximation import laplace_approximation


@pytest.fixture
def normal_quantiles():
    """Quantiles drawn from N(0, 1) for exact-posterior benchmarks."""
    from scipy.stats import norm
    return ElicitedQuantiles(
        lower=norm.ppf(0.05),
        q1=norm.ppf(0.25),
        median=0.0,
        q3=norm.ppf(0.75),
        upper=norm.ppf(0.95),
    )


class TestModeWithinCI:
    """The posterior mode must lie within the Laplace 95% CI."""

    def test_mode_in_laplace_ci(self, normal_quantiles):
        result = laplace_approximation(normal_quantiles, y=0.5, sigma_y=1.0)
        lo, hi = result["laplace_ci"]
        mode = result["mode"]
        assert lo <= mode <= hi, f"mode={mode} not in CI=[{lo}, {hi}]"

    def test_mode_in_skew_corrected_ci(self, symmetric_quantiles):
        result = laplace_approximation(symmetric_quantiles, y=0.0, sigma_y=1.0)
        lo, hi = result["skew_corrected_ci"]
        mode = result["mode"]
        assert lo <= mode <= hi


class TestLaplaceSD:
    """Laplace SD must be positive and finite."""

    def test_sd_positive(self, normal_quantiles):
        result = laplace_approximation(normal_quantiles, y=0.0, sigma_y=1.0)
        assert result["laplace_sd"] > 0
        assert math.isfinite(result["laplace_sd"])


class TestKLNonnegative:
    """KL(Laplace || quadrature) must be >= 0."""

    def test_kl_nonneg(self, normal_quantiles):
        result = laplace_approximation(normal_quantiles, y=0.0, sigma_y=1.0)
        assert result["kl_vs_quadrature"] >= 0.0


class TestQuadratureMoments:
    """Quadrature mean and SD must be finite."""

    def test_quadrature_finite(self, symmetric_quantiles):
        result = laplace_approximation(symmetric_quantiles, y=0.0, sigma_y=1.0)
        assert math.isfinite(result["quadrature_mean"])
        assert math.isfinite(result["quadrature_sd"])
        assert result["quadrature_sd"] > 0
