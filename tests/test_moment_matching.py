"""Tests for moment matching / GMM prior fitting."""

import numpy as np
import pytest

from priorlab.moment_matching import fit_gmm, _raw_moments_from_quantiles
from priorlab.models import ElicitedQuantiles


def test_j_stat_nonnegative(symmetric_quantiles):
    """Hansen's J-statistic must be >= 0 for all families."""
    result = fit_gmm(symmetric_quantiles)
    for fam, info in result["gmm_fits"].items():
        assert info["j_stat"] >= 0, f"J-stat negative for {fam}: {info['j_stat']}"


def test_moments_ordering(symmetric_quantiles):
    """E[X^2] >= E[X]^2 (variance non-negative)."""
    result = fit_gmm(symmetric_quantiles)
    m = result["moment_estimates"]
    assert m[1] >= m[0] ** 2 - 1e-6, "m2 < m1^2 implies negative variance"


def test_gmm_returns_all_keys(symmetric_quantiles):
    """Result dict must contain the five specified keys."""
    result = fit_gmm(symmetric_quantiles)
    for key in ["moment_estimates", "gmm_fits", "best_gmm_family",
                "kl_gmm_vs_quantile", "overall_best_method"]:
        assert key in result, f"Missing key: {key}"


def test_positive_quantiles_fit_gamma(skewed_quantiles):
    """Positive-only skewed quantiles should produce a gamma GMM fit."""
    result = fit_gmm(skewed_quantiles)
    assert "gamma" in result["gmm_fits"], "Gamma family missing for positive quantiles"
    assert result["gmm_fits"]["gamma"]["j_pvalue"] >= 0


def test_overall_best_method_valid(symmetric_quantiles):
    """overall_best_method must be 'gmm' or 'quantile_matching'."""
    result = fit_gmm(symmetric_quantiles)
    assert result["overall_best_method"] in ("gmm", "quantile_matching")
