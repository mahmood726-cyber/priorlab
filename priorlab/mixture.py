"""Mixture prior construction via EM algorithm.

Fit multi-component Gaussian mixture distributions for multi-modal
expert beliefs.  Includes model selection via BIC and K-means++
initialization.
"""

import numpy as np
from scipy import stats


def _kmeans_plus_plus(grid, density, k, rng):
    """K-means++ initialization: pick K centers weighted by density.

    First center chosen with probability proportional to density.
    Subsequent centers chosen with probability proportional to
    squared distance from the nearest existing center.
    """
    density = np.asarray(density, dtype=float)
    density = np.maximum(density, 0.0)
    total = density.sum()
    if total <= 0:
        # Fallback: uniform spacing
        idx = np.linspace(0, len(grid) - 1, k, dtype=int)
        return grid[idx]

    probs = density / total
    centers = []

    # First center: sample weighted by density
    idx = rng.choice(len(grid), p=probs)
    centers.append(grid[idx])

    for _ in range(1, k):
        # Squared distances from nearest existing center
        dists = np.array([min((grid[i] - c) ** 2 for c in centers)
                          for i in range(len(grid))])
        d_probs = dists / dists.sum() if dists.sum() > 0 else probs
        idx = rng.choice(len(grid), p=d_probs)
        centers.append(grid[idx])

    return np.array(centers)


def _em_fit(grid, density, k, max_iter=100, tol=1e-6, seed=42):
    """EM algorithm for K-component Normal mixture on a density grid.

    The density values are treated as weights at grid points.

    Parameters
    ----------
    grid : ndarray
        Evaluation grid.
    density : ndarray
        Density values (non-negative, need not integrate to 1).
    k : int
        Number of components.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.
    seed : int
        Random seed for initialization.

    Returns
    -------
    dict with weights, means, variances, log_likelihood, converged, iterations.
    """
    rng = np.random.default_rng(seed)
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    density = np.maximum(density, 0.0)
    n = len(grid)

    # Normalize density as weights
    w_total = density.sum()
    if w_total <= 0:
        w_total = 1.0
    weights_data = density / w_total  # weight of each grid point

    # Initialize via K-means++
    centers = _kmeans_plus_plus(grid, density, k, rng)
    span = grid[-1] - grid[0]
    init_var = (span / (2 * k)) ** 2
    init_var = max(init_var, 1e-6)

    pi = np.ones(k) / k
    mu = centers.copy()
    var = np.full(k, init_var)

    prev_ll = -np.inf
    converged = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1

        # E-step: compute responsibilities gamma[j, i]
        gamma = np.zeros((k, n))
        for j in range(k):
            sigma_j = np.sqrt(max(var[j], 1e-12))
            gamma[j] = pi[j] * stats.norm.pdf(grid, loc=mu[j], scale=sigma_j)

        gamma_sum = gamma.sum(axis=0)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma /= gamma_sum  # gamma[j, i] = responsibility of component j for point i

        # Weight gamma by data density
        weighted_gamma = gamma * weights_data[np.newaxis, :]  # (k, n)

        # M-step
        nk = weighted_gamma.sum(axis=1)  # effective count per component
        nk = np.maximum(nk, 1e-12)

        for j in range(k):
            mu[j] = np.sum(weighted_gamma[j] * grid) / nk[j]
            var[j] = np.sum(weighted_gamma[j] * (grid - mu[j]) ** 2) / nk[j]
            var[j] = max(var[j], 1e-6)

        pi = nk / nk.sum()

        # Log-likelihood
        mix_density = np.zeros(n)
        for j in range(k):
            sigma_j = np.sqrt(max(var[j], 1e-12))
            mix_density += pi[j] * stats.norm.pdf(grid, loc=mu[j], scale=sigma_j)
        mix_density = np.maximum(mix_density, 1e-300)
        ll = np.sum(weights_data * np.log(mix_density))

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    # Sort components by mean
    order = np.argsort(mu)
    mu = mu[order]
    var = var[order]
    pi = pi[order]

    # Final mixture PDF on grid
    mix_pdf = np.zeros(n)
    for j in range(k):
        sigma_j = np.sqrt(max(var[j], 1e-12))
        mix_pdf += pi[j] * stats.norm.pdf(grid, loc=mu[j], scale=sigma_j)

    return {
        "weights": pi.tolist(),
        "means": mu.tolist(),
        "variances": var.tolist(),
        "log_likelihood": float(ll),
        "converged": converged,
        "iterations": iterations,
        "mixture_pdf": mix_pdf.tolist(),
    }


def _bic(ll, k_params, n_eff):
    """BIC = -2*LL + k_params * log(n_eff)."""
    return -2.0 * ll + k_params * np.log(max(n_eff, 2.0))


def _effective_sample_size(density):
    """Effective sample size from normalized density weights.

    N_eff = 1 / sum(w_i^2) where w_i = density_i / sum(density).
    For uniform density, N_eff = N.  For peaked density, N_eff << N.
    This prevents grid resolution from inflating BIC penalty.
    """
    d = np.asarray(density, dtype=float)
    d = np.maximum(d, 0.0)
    total = d.sum()
    if total <= 0:
        return 2.0
    w = d / total
    ss = np.sum(w ** 2)
    if ss <= 0:
        return 2.0
    return 1.0 / ss


def fit_mixture(grid, density, max_k=3, max_iter=100, tol=1e-6, seed=42):
    """Fit Gaussian mixture with model selection via BIC.

    Tries K=1,2,...,max_k components and selects the model with
    lowest BIC.

    Parameters
    ----------
    grid : array-like
        Evaluation grid.
    density : array-like
        Density values on grid.
    max_k : int
        Maximum number of components to try.
    max_iter : int
        Maximum EM iterations per fit.
    tol : float
        EM convergence tolerance.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: n_components, weights, means, variances,
    bic_scores, log_likelihood, converged, iterations, mixture_pdf.
    """
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    n_eff = _effective_sample_size(density)

    best_bic = np.inf
    best_result = None
    bic_scores = {}

    for k in range(1, max_k + 1):
        result = _em_fit(grid, density, k, max_iter=max_iter, tol=tol, seed=seed)
        # K_params = 3K - 1 (K means + K variances + K-1 weights)
        k_params = 3 * k - 1
        # Scale weighted LL by N_eff so BIC penalty is commensurate
        scaled_ll = result["log_likelihood"] * n_eff
        b = _bic(scaled_ll, k_params, n_eff)
        bic_scores[k] = float(b)

        if b < best_bic:
            best_bic = b
            best_result = result
            best_k = k

    return {
        "n_components": best_k,
        "weights": best_result["weights"],
        "means": best_result["means"],
        "variances": best_result["variances"],
        "bic_scores": bic_scores,
        "log_likelihood": best_result["log_likelihood"],
        "converged": best_result["converged"],
        "iterations": best_result["iterations"],
        "mixture_pdf": best_result["mixture_pdf"],
    }
