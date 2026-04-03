import numpy as np
import pytest

from priorlab.information_geometry import (
    fisher_metric,
    fisher_rao_distance,
    geodesic_path,
    information_geometry,
    natural_gradient,
    pairwise_geodesic_matrix,
    scalar_curvature,
)


def test_geodesic_distance_symmetric():
    """Fisher-Rao distance must be symmetric: d(A,B) == d(B,A)."""
    d_ab = fisher_rao_distance(0, 1, 2, 1.5)
    d_ba = fisher_rao_distance(2, 1.5, 0, 1)
    assert abs(d_ab - d_ba) < 1e-12


def test_geodesic_distance_self_is_zero():
    """Distance from a distribution to itself must be 0."""
    d = fisher_rao_distance(1.0, 2.0, 1.0, 2.0)
    assert abs(d) < 1e-12


def test_pairwise_matrix_shape():
    """Pairwise distance matrix for 3 experts should be 3x3."""
    experts = [
        {"mu": 0, "sigma": 1},
        {"mu": 1, "sigma": 1},
        {"mu": 0, "sigma": 2},
    ]
    mat = pairwise_geodesic_matrix(experts)
    assert mat.shape == (3, 3)
    # Diagonal is zero
    for i in range(3):
        assert abs(mat[i, i]) < 1e-12
    # Symmetric
    for i in range(3):
        for j in range(3):
            assert abs(mat[i, j] - mat[j, i]) < 1e-12


def test_scalar_curvature_is_negative_half():
    """Normal manifold has constant scalar curvature -1/2."""
    assert scalar_curvature() == -0.5


def test_geodesic_path_endpoints():
    """Geodesic path must start at A and end at B."""
    path = geodesic_path(0, 1, 3, 2, n_steps=20)
    assert len(path) == 20
    # Start
    assert abs(path[0]["mu"] - 0) < 0.01
    assert abs(path[0]["sigma"] - 1) < 0.01
    # End
    assert abs(path[-1]["mu"] - 3) < 0.01
    assert abs(path[-1]["sigma"] - 2) < 0.01
    # t values monotonically increase
    t_vals = [p["t"] for p in path]
    assert t_vals == sorted(t_vals)
