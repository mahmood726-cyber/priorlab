"""Kernel Stein Discrepancy for prior goodness-of-fit.

Measures how well a fitted parametric distribution matches the elicited
quantiles using the kernel Stein discrepancy (KSD) with an RBF kernel
and score functions for Normal, LogNormal, and Gamma families.
"""

import numpy as np
from priorlab.fitting import fit_all_distributions
from priorlab.models import ElicitedQuantiles


# ---------------------------------------------------------------------------
# Score functions: s(x) = d/dx log p(x)
# ---------------------------------------------------------------------------

def _score_normal(x, mu, sigma):
    """Score function for Normal(mu, sigma^2)."""
    return -(x - mu) / (sigma ** 2)


def _score_lognormal(x, mu, sigma):
    """Score function for LogNormal(mu, sigma^2).

    p(x) = (1/(x*sigma*sqrt(2pi))) * exp(-(log(x)-mu)^2/(2*sigma^2))
    s(x) = -(log(x) - mu) / (sigma^2 * x) - 1/x
    """
    x_safe = np.maximum(x, 1e-300)
    return -(np.log(x_safe) - mu) / (sigma ** 2 * x_safe) - 1.0 / x_safe


def _score_gamma(x, alpha, beta):
    """Score function for Gamma(alpha, beta) with pdf x^{a-1} e^{-x/beta} / (beta^a Gamma(a)).

    s(x) = (alpha - 1)/x - 1/beta
    """
    x_safe = np.maximum(x, 1e-300)
    return (alpha - 1.0) / x_safe - 1.0 / beta


def _get_score_fn(family, params):
    """Return the score function for a fitted distribution family."""
    if family == "normal":
        mu, sigma = params["mu"], params["sigma"]
        return lambda x: _score_normal(x, mu, sigma)
    elif family == "lognormal":
        mu, sigma = params["mu_log"], params["sigma_log"]
        return lambda x: _score_lognormal(x, mu, sigma)
    elif family == "gamma":
        alpha, beta = params["shape"], params["scale"]
        return lambda x: _score_gamma(x, alpha, beta)
    elif family == "t":
        mu, sigma = params["loc"], params["scale"]
        return lambda x: _score_normal(x, mu, sigma)
    elif family == "halfcauchy":
        s = params["scale"]
        return lambda x: -2.0 * x / (s ** 2 + np.maximum(x, 1e-300) ** 2)
    elif family == "beta":
        a, b = params["alpha"], params["beta"]
        return lambda x: (a - 1.0) / np.maximum(x, 1e-300) - (b - 1.0) / np.maximum(1.0 - x, 1e-300)
    else:
        return lambda x: np.zeros_like(x)


# ---------------------------------------------------------------------------
# Vectorized KSD computation
# ---------------------------------------------------------------------------

def _restrict_to_support(grid, family):
    """Restrict the evaluation grid to a family's valid support.

    The score functions for positive-support families (``lognormal``,
    ``gamma``, ``halfcauchy``) and the unit-interval ``beta`` family diverge
    as x approaches the boundary of their support (e.g. ``(alpha-1)/x`` as
    ``x -> 0``).  Evaluating the score on a grid that crosses the boundary
    produces overflow and a non-finite KSD.  Clip the grid to the interior of
    the support so the discrepancy stays finite and comparable across families.
    """
    grid = np.asarray(grid, dtype=float)
    if family in ("lognormal", "gamma", "halfcauchy"):
        sel = grid > 0
    elif family == "beta":
        sel = (grid > 0) & (grid < 1)
    else:
        return grid
    restricted = grid[sel]
    return restricted if restricted.size >= 2 else grid


def _median_heuristic(samples):
    """Bandwidth via median heuristic (vectorized)."""
    samples = np.asarray(samples)
    # Pairwise distances via broadcasting
    diffs = np.abs(samples[:, None] - samples[None, :])
    # Upper triangle (exclude diagonal)
    idx = np.triu_indices(len(samples), k=1)
    pairwise = diffs[idx]
    if len(pairwise) == 0:
        return 1.0
    med = float(np.median(pairwise))
    return max(med, 1e-6)


def _compute_ksd_vectorized(samples, score_fn):
    """V-statistic KSD^2 fully vectorized with numpy broadcasting.

    KSD^2 = (1/n^2) * sum_{i,j} u_p(x_i, x_j)
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    h = _median_heuristic(samples)
    h2 = h ** 2
    h4 = h ** 4

    # Score values for all samples. Clamp to a large finite magnitude so that
    # samples landing near a distribution's support boundary (where the score
    # legitimately diverges) cannot overflow the squared score products below.
    scores = score_fn(samples)  # shape (n,)
    scores = np.clip(np.nan_to_num(scores, nan=0.0), -1e150, 1e150)

    # Pairwise differences: diff[i,j] = x_i - x_j
    xi = samples[:, None]  # (n, 1)
    xj = samples[None, :]  # (1, n)
    diff = xi - xj  # (n, n)

    # RBF kernel matrix
    k_mat = np.exp(-(diff ** 2) / (2.0 * h2))  # (n, n)

    # Score outer product
    si = scores[:, None]  # (n, 1)
    sj = scores[None, :]  # (1, n)

    # Stein kernel components
    term1 = si * sj * k_mat
    term2 = si * (diff / h2) * k_mat       # s(x_i) * dk/dy = s(x_i) * -(y-x)/h^2 * k = s(x_i) * (x-y)/h^2 * k
    term3 = sj * (-diff / h2) * k_mat      # s(x_j) * dk/dx = s(x_j) * -(x-y)/h^2 * k
    term4 = (1.0 / h2 - diff ** 2 / h4) * k_mat  # d^2k/dxdy

    stein_mat = term1 + term2 + term3 + term4
    ksd_sq = float(np.sum(stein_mat)) / (n * n)
    return max(ksd_sq, 0.0)


def _bootstrap_p_value(samples, score_fn, observed_ksd_sq, n_perm=199):
    """Bootstrap p-value via wild bootstrap (vectorized KSD)."""
    rng = np.random.RandomState(42)
    count = 0
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_samples = samples * signs
        perm_ksd = _compute_ksd_vectorized(perm_samples, score_fn)
        if perm_ksd >= observed_ksd_sq:
            count += 1
    return (count + 1) / (n_perm + 1)


# Keep scalar versions for direct testing
def _stein_kernel(xi, xj, score_fn, h):
    """Stein kernel u_p(x_i, x_j) - scalar version."""
    k = np.exp(-((xi - xj) ** 2) / (2.0 * h ** 2))
    si = score_fn(np.asarray(xi))
    sj = score_fn(np.asarray(xj))
    diff = xi - xj
    h2 = h ** 2
    dk_dy = diff / h2 * k
    dk_dx = -diff / h2 * k
    d2k_dxdy = (1.0 / h2 - diff ** 2 / (h2 ** 2)) * k
    return float(si * sj * k + si * dk_dy + sj * dk_dx + d2k_dxdy)


def _compute_ksd(samples, score_fn):
    """V-statistic KSD^2 (delegates to vectorized version)."""
    return _compute_ksd_vectorized(samples, score_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stein_discrepancy(quantiles: ElicitedQuantiles, n_grid: int = 200):
    """Compute Kernel Stein Discrepancy for all fitted families.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        The expert's elicited quantiles.
    n_grid : int
        Number of evaluation grid points (default 200).

    Returns
    -------
    dict with keys:
        ksd_by_family : dict[str, float] - KSD for each family
        best_family_ksd : str - family with lowest KSD
        p_values : dict[str, float] - bootstrap p-values
        overall_best_fit : str - same as best_family_ksd
    """
    fits = fit_all_distributions(quantiles)

    # Generate evaluation grid
    span = quantiles.upper - quantiles.lower
    lo = quantiles.lower - 0.1 * span
    hi = quantiles.upper + 0.1 * span
    grid = np.linspace(lo, hi, n_grid)

    ksd_by_family = {}
    p_values = {}

    for fitted in fits:
        family = fitted.family
        if fitted.ks_distance > 900:
            continue  # Skip failed fits

        score_fn = _get_score_fn(family, fitted.params)
        fam_grid = _restrict_to_support(grid, family)
        ksd_sq = _compute_ksd_vectorized(fam_grid, score_fn)
        ksd_val = float(np.sqrt(ksd_sq))
        ksd_by_family[family] = ksd_val

        # Bootstrap p-value
        p_val = _bootstrap_p_value(fam_grid, score_fn, ksd_sq, n_perm=199)
        p_values[family] = p_val

    if not ksd_by_family:
        return {
            "ksd_by_family": {},
            "best_family_ksd": None,
            "p_values": {},
            "overall_best_fit": None,
        }

    best = min(ksd_by_family, key=ksd_by_family.get)
    return {
        "ksd_by_family": ksd_by_family,
        "best_family_ksd": best,
        "p_values": p_values,
        "overall_best_fit": best,
    }
