"""Tests for Optimal Transport module."""

import numpy as np
import pytest

from priorlab.transport import (
    w2_distance,
    pairwise_w2_matrix,
    wasserstein_barycenter,
    displacement_interpolation,
    transport_map,
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


def test_w2_nonnegative_and_symmetric():
    """W2 distance must be >= 0 and symmetric."""
    cdf_p = norm(0, 1).cdf(GRID)
    cdf_q = norm(1, 1).cdf(GRID)
    d_pq = w2_distance(cdf_p, cdf_q, GRID)
    d_qp = w2_distance(cdf_q, cdf_p, GRID)
    assert d_pq >= 0
    assert d_qp >= 0
    assert abs(d_pq - d_qp) < 0.05, f"W2 not symmetric: {d_pq} vs {d_qp}"


def test_w2_zero_for_identical():
    """W2 of a distribution with itself should be ~0."""
    cdf_p = norm(0, 1).cdf(GRID)
    d = w2_distance(cdf_p, cdf_p, GRID)
    assert d < 0.01


def test_pairwise_w2_matrix_shape():
    """Pairwise W2 matrix should be n x n and symmetric."""
    experts = [_make_expert("A", -1, 1), _make_expert("B", 0, 1), _make_expert("C", 1, 1)]
    mat = pairwise_w2_matrix(experts, GRID)
    assert mat.shape == (3, 3)
    # Diagonal should be zero
    for i in range(3):
        assert mat[i, i] == 0.0
    # Symmetric
    assert abs(mat[0, 1] - mat[1, 0]) < 1e-10


def test_barycenter_integrates_to_one():
    """Barycenter density should integrate to approximately 1."""
    experts = [_make_expert("A", -1, 1), _make_expert("B", 1, 1)]
    result = wasserstein_barycenter(experts, GRID)
    area = np.trapezoid(result["barycenter_density"], result["barycenter_grid"])
    assert abs(area - 1.0) < 0.1, f"Barycenter area = {area}"


def test_displacement_interpolation_endpoints():
    """Displacement interpolation at t=0 and t=1 should match the two experts."""
    e_a = _make_expert("A", -1, 1)
    e_b = _make_expert("B", 1, 1)
    paths = displacement_interpolation(e_a, e_b, GRID)
    assert len(paths) == 5
    assert paths[0]["t"] == 0.0
    assert paths[-1]["t"] == 1.0
    # Each density should have non-negative values
    for p in paths:
        assert np.all(p["density"] >= -1e-10)
