"""Tests for Statistical Depth Functions module."""

import numpy as np
import pytest

from priorlab.depth_functions import (
    tukey_depth,
    simplicial_depth,
    zonoid_depth,
    depth_analysis,
    expert_depths,
)
from priorlab.models import ExpertPrior, ElicitedQuantiles, FittedDistribution
from scipy.stats import norm


def _make_expert(expert_id, mu, sigma):
    """Create an ExpertPrior with a Normal best fit."""
    eq = ElicitedQuantiles(
        lower=mu - 2 * sigma,
        q1=mu - 0.674 * sigma,
        median=mu,
        q3=mu + 0.674 * sigma,
        upper=mu + 2 * sigma,
    )
    fitted = FittedDistribution(
        family="normal",
        params={"mu": mu, "sigma": sigma},
        ks_distance=0.01,
    )
    return ExpertPrior(
        expert_id=expert_id, label=f"Expert {expert_id}",
        quantiles=eq, best_fit=fitted,
    )


GRID = np.linspace(-5, 5, 500)
CDF_NORMAL = norm(0, 1).cdf(GRID)


def test_tukey_in_range():
    """Tukey depth must be in [0, 0.5]."""
    td = tukey_depth(GRID, CDF_NORMAL)
    assert np.all(td >= 0.0)
    assert np.all(td <= 0.5 + 1e-10)


def test_simplicial_in_range():
    """Simplicial depth must be in [0, 0.5]."""
    sd = simplicial_depth(GRID, CDF_NORMAL)
    assert np.all(sd >= 0.0)
    assert np.all(sd <= 0.5 + 1e-10)


def test_zonoid_in_range():
    """Zonoid depth must be in [0, 1]."""
    zd = zonoid_depth(GRID, CDF_NORMAL)
    assert np.all(zd >= -1e-10)
    assert np.all(zd <= 1.0 + 1e-10)


def test_expert_depths_central_higher():
    """A central expert (median=0) should have higher depth than an extreme one."""
    experts = [
        _make_expert("Central", 0.0, 1.0),
        _make_expert("Extreme", 3.0, 1.0),
    ]
    depths = expert_depths(experts, GRID)
    assert depths["Central"] > depths["Extreme"]


def test_depth_analysis_full():
    """Full depth analysis should return all expected keys."""
    experts = [
        _make_expert("A", -1, 1),
        _make_expert("B", 0, 1),
        _make_expert("C", 1, 1),
    ]
    result = depth_analysis(experts, GRID)
    assert "tukey_depth_grid" in result
    assert "simplicial_depth_grid" in result
    assert "expert_depths" in result
    assert "depth_ranking" in result
    assert "depth_contours" in result
    assert "trimmed_aggregation" in result
    # Tukey depth grid should be correct length
    assert len(result["tukey_depth_grid"]) == len(GRID)
    # Depth ranking should have 3 entries
    assert len(result["depth_ranking"]) == 3
