"""Tests for priorlab.prior_predictive module."""

import math
import numpy as np
import pytest

from priorlab.models import ElicitedQuantiles
from priorlab.prior_predictive import prior_predictive_check


class TestPredictiveSD:
    """Predictive SD must be strictly positive."""

    def test_sd_positive(self, symmetric_quantiles):
        result = prior_predictive_check(symmetric_quantiles, sigma_y=1.0)
        assert result["predictive_sd"] > 0


class TestConflictIndex:
    """Conflict index must be in [0, 1)."""

    def test_conflict_bounded(self, symmetric_quantiles):
        result = prior_predictive_check(symmetric_quantiles, sigma_y=1.0, y_obs=5.0)
        ci = result["conflict_index"]
        assert 0.0 <= ci < 1.0, f"conflict_index={ci} out of [0,1)"


class TestPlausibleFraction:
    """Plausible fraction must be in [0, 1]."""

    def test_fraction_in_unit(self, symmetric_quantiles):
        result = prior_predictive_check(
            symmetric_quantiles, sigma_y=1.0, plausible_range=(-5, 5),
        )
        frac = result["plausible_fraction"]
        assert 0.0 <= frac <= 1.0


class TestHistogramData:
    """Histogram must have valid structure."""

    def test_histogram_shape(self, symmetric_quantiles):
        result = prior_predictive_check(symmetric_quantiles, sigma_y=1.0, n_bins=30)
        assert len(result["histogram_values"]) == 30
        # bins edges = n_bins + 1
        assert len(result["histogram_bins"]) == 31


class TestPredictivePValue:
    """Prior predictive p-value must be in [0, 1] when observed data given."""

    def test_p_in_unit(self, symmetric_quantiles):
        result = prior_predictive_check(
            symmetric_quantiles, sigma_y=1.0, y_obs=0.0,
        )
        p = result["prior_predictive_p"]
        assert 0.0 <= p <= 1.0
