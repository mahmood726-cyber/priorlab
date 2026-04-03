import numpy as np
import pytest
from scipy.stats import norm

from priorlab.divergence import (
    hellinger_distance,
    kl_divergence,
    pairwise_divergences,
    total_variation,
    wasserstein_1,
)


def _make_pdf(mu, sigma, grid):
    return norm(loc=mu, scale=sigma).pdf(grid)


GRID = np.linspace(-5, 5, 500)


def test_hellinger_in_unit_interval():
    """Hellinger distance must be in [0, 1]."""
    p = _make_pdf(0, 1, GRID)
    q = _make_pdf(1, 1, GRID)
    h = hellinger_distance(p, q, GRID)
    assert 0 <= h <= 1


def test_hellinger_symmetric():
    """Hellinger must be symmetric: H(p,q) == H(q,p)."""
    p = _make_pdf(0, 0.5, GRID)
    q = _make_pdf(1, 1.5, GRID)
    assert abs(hellinger_distance(p, q, GRID) - hellinger_distance(q, p, GRID)) < 1e-12


def test_kl_zero_for_identical():
    """KL(p||p) should be zero (or very near zero)."""
    p = _make_pdf(0, 1, GRID)
    kl = kl_divergence(p, p, GRID)
    assert abs(kl) < 1e-10


def test_wasserstein_positive_for_shifted():
    """W1 between two shifted normals should be roughly |mu1 - mu2|."""
    p = _make_pdf(0, 1, GRID)
    q = _make_pdf(2, 1, GRID)
    w = wasserstein_1(p, q, GRID)
    assert w > 0
    # For identical-variance normals, W1 ~ |mu1-mu2| = 2
    assert abs(w - 2.0) < 0.15


def test_pairwise_matrices_shape_and_disagreement():
    """Pairwise divergences for 3 experts should produce 3x3 matrices."""
    pdfs = [_make_pdf(mu, 1, GRID) for mu in [-1, 0, 1]]
    result = pairwise_divergences(pdfs, GRID)
    assert result["kl_matrix"].shape == (3, 3)
    assert result["hellinger_matrix"].shape == (3, 3)
    assert result["wasserstein_matrix"].shape == (3, 3)
    assert result["tv_matrix"].shape == (3, 3)
    # Diagonal should be zero
    for key in ("kl_matrix", "hellinger_matrix", "wasserstein_matrix", "tv_matrix"):
        for i in range(3):
            assert result[key][i, i] == 0.0
    # Disagreement index > 0 for different distributions
    assert result["disagreement_index"] > 0
