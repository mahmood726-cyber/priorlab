"""Statistical divergences between expert priors.

Information-theoretic measures of disagreement between experts,
all computed on discrete density grids.
"""

import numpy as np


def _ensure_positive(p, epsilon=1e-300):
    """Clamp density to positive values to avoid log(0)."""
    return np.maximum(np.asarray(p, dtype=float), epsilon)


def kl_divergence(p, q, grid, epsilon=1e-300):
    """KL(p||q) = sum(p * log(p/q)) * dx.  Asymmetric."""
    p = _ensure_positive(p, epsilon)
    q = _ensure_positive(q, epsilon)
    dx = np.diff(grid)
    # Use midpoint values for the integrand
    p_mid = 0.5 * (p[:-1] + p[1:])
    q_mid = 0.5 * (q[:-1] + q[1:])
    p_mid = np.maximum(p_mid, epsilon)
    q_mid = np.maximum(q_mid, epsilon)
    integrand = p_mid * np.log(p_mid / q_mid)
    return float(np.sum(integrand * dx))


def symmetric_kl(p, q, grid, epsilon=1e-300):
    """Jeffreys divergence: J(p,q) = KL(p||q) + KL(q||p)."""
    return kl_divergence(p, q, grid, epsilon) + kl_divergence(q, p, grid, epsilon)


def hellinger_distance(p, q, grid):
    """Hellinger distance H(p,q) = sqrt(1 - sum(sqrt(p*q))*dx).  In [0,1], symmetric."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    bc = np.trapezoid(np.sqrt(p * q), grid)
    # Bhattacharyya coefficient should be in [0, 1]; clamp for numerical safety
    bc = min(max(bc, 0.0), 1.0)
    return float(np.sqrt(1.0 - bc))


def wasserstein_1(p, q, grid):
    """Wasserstein-1 (earth mover's) distance: W1 = integral |F(x)-G(x)| dx."""
    from scipy.integrate import cumulative_trapezoid
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Compute CDFs via cumulative trapezoid integration
    cdf_p = np.concatenate(([0.0], cumulative_trapezoid(p, grid)))
    cdf_q = np.concatenate(([0.0], cumulative_trapezoid(q, grid)))
    return float(np.trapezoid(np.abs(cdf_p - cdf_q), grid))


def total_variation(p, q, grid):
    """Total variation distance: TV = 0.5 * sum(|p - q|) * dx.  In [0,1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(0.5 * np.trapezoid(np.abs(p - q), grid))


def pairwise_divergences(pdfs, grid):
    """Compute pairwise divergence matrices for N expert density grids.

    Parameters
    ----------
    pdfs : list of array-like
        N density arrays, each evaluated on *grid*.
    grid : array-like
        Shared evaluation grid.

    Returns
    -------
    dict with keys: kl_matrix, hellinger_matrix, wasserstein_matrix,
    tv_matrix, disagreement_index.
    """
    n = len(pdfs)
    grid = np.asarray(grid, dtype=float)
    kl_mat = np.zeros((n, n))
    hell_mat = np.zeros((n, n))
    wass_mat = np.zeros((n, n))
    tv_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            kl_mat[i, j] = kl_divergence(pdfs[i], pdfs[j], grid)
            hell_mat[i, j] = hellinger_distance(pdfs[i], pdfs[j], grid)
            wass_mat[i, j] = wasserstein_1(pdfs[i], pdfs[j], grid)
            tv_mat[i, j] = total_variation(pdfs[i], pdfs[j], grid)

    # Disagreement index: mean pairwise Hellinger distance (upper triangle)
    if n >= 2:
        upper = hell_mat[np.triu_indices(n, k=1)]
        disagreement = float(np.mean(upper))
    else:
        disagreement = 0.0

    return {
        "kl_matrix": kl_mat,
        "hellinger_matrix": hell_mat,
        "wasserstein_matrix": wass_mat,
        "tv_matrix": tv_mat,
        "disagreement_index": disagreement,
    }
