"""Functional Data Analysis for Prior Distributions.

Treats expert density functions as elements of L^2 function space,
enabling functional PCA, Karhunen-Loeve expansion, Frechet statistics,
and Modified Band Depth for outlier detection.
"""

import numpy as np
from itertools import combinations


def l2_inner_product(f, g, grid):
    """L^2 inner product: <f, g> = integral f(x)*g(x) dx."""
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    grid = np.asarray(grid, dtype=float)
    return float(np.trapezoid(f * g, grid))


def l2_norm(f, grid):
    """L^2 norm: ||f|| = sqrt(<f, f>)."""
    return float(np.sqrt(max(l2_inner_product(f, f, grid), 0.0)))


def l2_distance(f, g, grid):
    """L^2 distance: d(f, g) = ||f - g||."""
    diff = np.asarray(f, dtype=float) - np.asarray(g, dtype=float)
    return l2_norm(diff, grid)


def functional_pca(densities, grid, variance_threshold=0.95):
    """Functional PCA on a set of expert densities.

    Parameters
    ----------
    densities : dict of {expert_id: array-like}
        Expert density functions evaluated on *grid*.
    grid : array-like
        Shared evaluation grid.
    variance_threshold : float
        Fraction of variance to retain (default 0.95).

    Returns
    -------
    dict with keys: mean_density, eigenvalues, eigenfunctions, scores,
    variance_explained, n_components.
    """
    grid = np.asarray(grid, dtype=float)
    expert_ids = list(densities.keys())
    n = len(expert_ids)
    m = len(grid)

    # Build matrix of densities (n x m)
    D = np.zeros((n, m))
    for i, eid in enumerate(expert_ids):
        D[i] = np.asarray(densities[eid], dtype=float)

    # Mean density
    mean_density = D.mean(axis=0)

    # Center
    centered = D - mean_density[np.newaxis, :]

    # Covariance matrix discretized: C(s,t) = (1/n) * sum(centered_i(s) * centered_i(t))
    # Discretize as m x m matrix
    cov_matrix = (centered.T @ centered) / n

    # Weight by grid spacing for proper L^2 inner product
    dx = np.gradient(grid)
    sqrt_dx = np.sqrt(np.maximum(dx, 1e-15))
    weighted_cov = cov_matrix * np.outer(sqrt_dx, sqrt_dx)

    # Eigendecomposition (symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negative eigenvalues to zero (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Convert back to eigenfunctions by removing grid weighting
    eigenfunctions = eigenvectors / sqrt_dx[:, np.newaxis]

    # Variance explained
    total_var = eigenvalues.sum()
    if total_var > 0:
        var_explained = eigenvalues / total_var
        cumvar = np.cumsum(var_explained)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = min(n_components, len(eigenvalues))
    else:
        var_explained = np.zeros_like(eigenvalues)
        n_components = 1

    # Scores: project each centered density onto eigenfunctions
    scores = {}
    for i, eid in enumerate(expert_ids):
        s = []
        for j in range(n_components):
            score = l2_inner_product(centered[i], eigenfunctions[:, j], grid)
            s.append(score)
        scores[eid] = s

    return {
        "mean_density": mean_density.tolist(),
        "eigenvalues": eigenvalues[:n_components].tolist(),
        "eigenfunctions": [eigenfunctions[:, j].tolist() for j in range(n_components)],
        "scores": scores,
        "variance_explained": var_explained[:n_components].tolist(),
        "n_components": n_components,
    }


def karhunen_loeve_reconstruct(mean_density, eigenfunctions, scores):
    """Reconstruct a density via Karhunen-Loeve expansion.

    density(x) = mean(x) + sum_j(score_j * eigenfunction_j(x))

    Parameters
    ----------
    mean_density : array-like
        Mean density function.
    eigenfunctions : list of array-like
        Eigenfunctions from fPCA.
    scores : list of float
        Projection scores for one expert.

    Returns
    -------
    numpy array of reconstructed density.
    """
    result = np.asarray(mean_density, dtype=float).copy()
    for score, ef in zip(scores, eigenfunctions):
        result += score * np.asarray(ef, dtype=float)
    return result


def frechet_mean(densities, grid):
    """Frechet mean in L^2: the density minimizing sum of squared L^2 distances.

    For L^2, this equals the arithmetic mean.
    """
    grid = np.asarray(grid, dtype=float)
    arrs = [np.asarray(d, dtype=float) for d in densities.values()]
    return np.mean(arrs, axis=0)


def frechet_variance(densities, grid):
    """Frechet variance: mean squared L^2 distance to Frechet mean."""
    grid = np.asarray(grid, dtype=float)
    fm = frechet_mean(densities, grid)
    total = 0.0
    for d in densities.values():
        total += l2_distance(d, fm, grid) ** 2
    return total / len(densities)


def modified_band_depth(densities, grid):
    """Modified Band Depth for each expert density.

    For each density f_i, compute proportion of all pairs (f_j, f_k)
    whose band (pointwise min/max envelope) contains f_i at >= 50%
    of grid points.

    Parameters
    ----------
    densities : dict of {expert_id: array-like}
    grid : array-like

    Returns
    -------
    dict of {expert_id: float} with depth values in [0, 1].
    """
    grid = np.asarray(grid, dtype=float)
    expert_ids = list(densities.keys())
    n = len(expert_ids)
    m = len(grid)

    arrs = {eid: np.asarray(densities[eid], dtype=float) for eid in expert_ids}

    if n < 3:
        # Need at least 3 for meaningful depth; return 1.0 for all
        return {eid: 1.0 for eid in expert_ids}

    depths = {}
    pairs = list(combinations(expert_ids, 2))
    n_pairs = len(pairs)

    for eid in expert_ids:
        f_i = arrs[eid]
        count = 0
        for eid_j, eid_k in pairs:
            if eid_j == eid or eid_k == eid:
                continue
            band_lo = np.minimum(arrs[eid_j], arrs[eid_k])
            band_hi = np.maximum(arrs[eid_j], arrs[eid_k])
            inside = np.sum((f_i >= band_lo - 1e-12) & (f_i <= band_hi + 1e-12))
            proportion = inside / m
            if proportion >= 0.5:
                count += 1
        # Depth = fraction of valid pairs
        valid_pairs = n_pairs - (n - 1)  # exclude pairs containing eid
        depths[eid] = count / max(valid_pairs, 1)

    return depths


def functional_analysis(densities, grid, variance_threshold=0.95):
    """Full functional data analysis of expert prior densities.

    Parameters
    ----------
    densities : dict of {expert_id: array-like}
        Expert density functions evaluated on *grid*.
    grid : array-like
        Shared evaluation grid.
    variance_threshold : float
        Fraction of variance to retain for fPCA (default 0.95).

    Returns
    -------
    dict with keys: mean_density, eigenvalues, eigenfunctions, scores,
    variance_explained, frechet_variance, functional_depths, n_components,
    l2_distances.
    """
    grid = np.asarray(grid, dtype=float)
    expert_ids = list(densities.keys())
    n = len(expert_ids)

    # fPCA
    fpca = functional_pca(densities, grid, variance_threshold)

    # Frechet variance
    fvar = frechet_variance(densities, grid)

    # Functional depths
    depths = modified_band_depth(densities, grid)

    # Pairwise L2 distances
    l2_dists = {}
    for i, eid_i in enumerate(expert_ids):
        for j, eid_j in enumerate(expert_ids):
            if i < j:
                d = l2_distance(densities[eid_i], densities[eid_j], grid)
                l2_dists[f"{eid_i}_vs_{eid_j}"] = d

    return {
        "mean_density": fpca["mean_density"],
        "eigenvalues": fpca["eigenvalues"],
        "eigenfunctions": fpca["eigenfunctions"],
        "scores": fpca["scores"],
        "variance_explained": fpca["variance_explained"],
        "frechet_variance": fvar,
        "functional_depths": depths,
        "n_components": fpca["n_components"],
        "l2_distances": l2_dists,
    }
