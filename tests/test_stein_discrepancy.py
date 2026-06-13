"""Tests for Kernel Stein Discrepancy module."""

import numpy as np
import pytest

from priorlab.stein_discrepancy import (
    stein_discrepancy,
    _compute_ksd,
    _score_normal,
    _median_heuristic,
    _stein_kernel,
)
from priorlab.models import ElicitedQuantiles


def test_ksd_nonnegative(symmetric_quantiles):
    """KSD must be >= 0 for all fitted families."""
    result = stein_discrepancy(symmetric_quantiles, n_grid=200)
    for family, ksd in result["ksd_by_family"].items():
        assert ksd >= 0, f"KSD for {family} is negative: {ksd}"


def test_ksd_returns_best_family(symmetric_quantiles):
    """Result must include best_family_ksd and overall_best_fit."""
    result = stein_discrepancy(symmetric_quantiles, n_grid=200)
    assert result["best_family_ksd"] is not None
    assert result["overall_best_fit"] is not None
    assert result["best_family_ksd"] == result["overall_best_fit"]
    # Best must be in the ksd_by_family dict
    assert result["best_family_ksd"] in result["ksd_by_family"]


def test_ksd_p_values_in_range(symmetric_quantiles):
    """Bootstrap p-values must be in (0, 1]."""
    result = stein_discrepancy(symmetric_quantiles, n_grid=200)
    for family, pval in result["p_values"].items():
        assert 0 < pval <= 1.0, f"p-value for {family} out of range: {pval}"


def test_ksd_skewed_quantiles(skewed_quantiles):
    """KSD should work for positive-only (skewed) quantiles."""
    result = stein_discrepancy(skewed_quantiles, n_grid=200)
    assert len(result["ksd_by_family"]) > 0
    for ksd in result["ksd_by_family"].values():
        assert ksd >= 0


def test_ksd_skewed_quantiles_all_finite(skewed_quantiles):
    """Positive-support families (lognormal/gamma/beta) must yield finite KSD.

    Regression: evaluating the score function on a grid crossing the support
    boundary previously overflowed to ``inf`` for these families.
    """
    result = stein_discrepancy(skewed_quantiles, n_grid=200)
    for fam, ksd in result["ksd_by_family"].items():
        assert np.isfinite(ksd), f"KSD for {fam} is not finite: {ksd}"


def test_score_normal_at_mean():
    """Normal score at mean should be zero."""
    mu, sigma = 2.0, 1.5
    score = _score_normal(mu, mu, sigma)
    assert abs(score) < 1e-12
