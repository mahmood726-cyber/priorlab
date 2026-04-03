import numpy as np
import pytest
from scipy.stats import norm

from priorlab.functional_bayes import (
    functional_analysis,
    functional_pca,
    l2_distance,
    l2_inner_product,
    l2_norm,
    modified_band_depth,
)

GRID = np.linspace(-5, 5, 300)


def _make_densities():
    """Three expert densities with different means."""
    return {
        "expert_A": norm(loc=-0.5, scale=0.8).pdf(GRID),
        "expert_B": norm(loc=0.0, scale=1.0).pdf(GRID),
        "expert_C": norm(loc=0.5, scale=0.8).pdf(GRID),
    }


def test_eigenvalues_nonnegative():
    """fPCA eigenvalues must be non-negative."""
    densities = _make_densities()
    result = functional_pca(densities, GRID)
    for ev in result["eigenvalues"]:
        assert ev >= -1e-12, f"Negative eigenvalue: {ev}"


def test_variance_explained_sums_to_one_or_less():
    """Variance explained fractions should sum to <= 1."""
    densities = _make_densities()
    result = functional_pca(densities, GRID)
    total = sum(result["variance_explained"])
    assert total <= 1.0 + 1e-10


def test_functional_depth_in_unit_interval():
    """Modified Band Depth values must be in [0, 1]."""
    densities = _make_densities()
    depths = modified_band_depth(densities, GRID)
    for eid, d in depths.items():
        assert 0.0 <= d <= 1.0 + 1e-10, f"Depth out of range for {eid}: {d}"


def test_l2_distance_triangle_inequality():
    """L2 distance must satisfy triangle inequality: d(A,C) <= d(A,B) + d(B,C)."""
    densities = _make_densities()
    d_ab = l2_distance(densities["expert_A"], densities["expert_B"], GRID)
    d_bc = l2_distance(densities["expert_B"], densities["expert_C"], GRID)
    d_ac = l2_distance(densities["expert_A"], densities["expert_C"], GRID)
    assert d_ac <= d_ab + d_bc + 1e-10


def test_full_analysis_returns_all_keys():
    """functional_analysis should return all expected keys."""
    densities = _make_densities()
    result = functional_analysis(densities, GRID)
    expected_keys = {
        "mean_density", "eigenvalues", "eigenfunctions", "scores",
        "variance_explained", "frechet_variance", "functional_depths",
        "n_components", "l2_distances",
    }
    assert expected_keys == set(result.keys())
    assert result["frechet_variance"] >= 0.0
    assert result["n_components"] >= 1
