"""Dirichlet Process Prior for Nonparametric Density Estimation.

When expert beliefs are multi-modal or irregular, parametric families
may not capture the shape.  A truncated Dirichlet Process (DP) prior
with stick-breaking construction (Sethuraman 1994) provides a flexible
nonparametric density estimate with uncertainty bands.

The base distribution G_0 is the best-fit parametric prior from fitting.py.
Atoms are drawn from G_0 and mixed via stick-breaking weights.  The DP
density is a kernel-smoothed mixture of atoms.
"""

import numpy as np
from scipy import stats

from priorlab.models import ElicitedQuantiles, FittedDistribution
from priorlab.fitting import fit_all_distributions, select_best_fit, _make_grid
from priorlab.divergence import kl_divergence


def _stick_breaking_weights(alpha, K, rng):
    """Generate stick-breaking weights (Sethuraman 1994).

    v_k ~ Beta(1, alpha) for k = 1, ..., K
    pi_k = v_k * prod(1 - v_j for j < k)

    Parameters
    ----------
    alpha : float
        Concentration parameter (>0).  Larger alpha => more components.
    K : int
        Truncation level.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of shape (K,) summing to ~1.
    """
    v = rng.beta(1.0, alpha, size=K)
    v[-1] = 1.0  # ensure weights sum to 1
    log_one_minus_v = np.log(np.maximum(1.0 - v, 1e-300))
    log_cum = np.concatenate(([0.0], np.cumsum(log_one_minus_v[:-1])))
    log_pi = np.log(np.maximum(v, 1e-300)) + log_cum
    pi = np.exp(log_pi)
    pi /= pi.sum()  # normalise for numerical safety
    return pi


def _silverman_bandwidth(sigma, n):
    """Silverman's rule of thumb: h = 1.06 * sigma * n^(-1/5)."""
    return 1.06 * sigma * max(n, 1) ** (-0.2)


def _sample_atoms(fitted, K, rng):
    """Draw K atoms from the base distribution G_0.

    Parameters
    ----------
    fitted : FittedDistribution
        Best-fit parametric distribution used as base measure.
    K : int
        Number of atoms.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of shape (K,)
    """
    family = fitted.family
    params = fitted.params

    if family == "normal":
        return rng.normal(params["mu"], params["sigma"], size=K)
    elif family == "lognormal":
        return rng.lognormal(params["mu_log"], params["sigma_log"], size=K)
    elif family == "gamma":
        return rng.gamma(params["shape"], scale=params["scale"], size=K)
    elif family == "beta":
        return rng.beta(params["alpha"], params["beta"], size=K)
    elif family == "t":
        return params["loc"] + params["scale"] * rng.standard_t(params["df"], size=K)
    elif family == "halfcauchy":
        # Half-Cauchy: |Cauchy(0, scale)|
        return np.abs(rng.standard_cauchy(size=K)) * params["scale"]
    else:
        # Fallback: normal with estimated moments from grid
        grid = np.array(fitted.x_grid)
        pdf = np.array(fitted.pdf_values)
        mu = np.trapezoid(grid * pdf, grid)
        var = np.trapezoid((grid - mu) ** 2 * pdf, grid)
        return rng.normal(mu, max(np.sqrt(var), 1e-6), size=K)


def _dp_density_on_grid(weights, atoms, grid, bandwidth):
    """Compute DP density: p(x) = sum_k pi_k * N(x | theta_k, h^2).

    Parameters
    ----------
    weights : np.ndarray of shape (K,)
    atoms : np.ndarray of shape (K,)
    grid : np.ndarray
    bandwidth : float

    Returns
    -------
    np.ndarray of shape (len(grid),)
    """
    density = np.zeros_like(grid, dtype=float)
    for w, theta in zip(weights, atoms):
        density += w * stats.norm.pdf(grid, loc=theta, scale=bandwidth)
    return density


def _base_sigma(fitted):
    """Estimate standard deviation from the fitted distribution parameters."""
    params = fitted.params
    family = fitted.family

    if family == "normal":
        return params["sigma"]
    elif family == "lognormal":
        s = params["sigma_log"]
        mu = params["mu_log"]
        return np.sqrt((np.exp(s ** 2) - 1) * np.exp(2 * mu + s ** 2))
    elif family == "gamma":
        return np.sqrt(params["shape"]) * params["scale"]
    elif family == "beta":
        a, b = params["alpha"], params["beta"]
        return np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    elif family == "t":
        return params["scale"]
    elif family == "halfcauchy":
        return params["scale"]
    else:
        # Fallback from grid
        grid = np.array(fitted.x_grid)
        pdf = np.array(fitted.pdf_values)
        mu = np.trapezoid(grid * pdf, grid)
        var = np.trapezoid((grid - mu) ** 2 * pdf, grid)
        return max(np.sqrt(var), 1e-6)


def dp_prior(quantiles, alpha=1.0, K_max=20, B=100, seed=42):
    """Dirichlet Process nonparametric density estimate for elicited quantiles.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        The five elicited quantile values.
    alpha : float
        DP concentration parameter (default 1.0).
        Larger => more components, closer to base distribution.
    K_max : int
        Truncation level for stick-breaking (default 20).
    B : int
        Number of bootstrap stick-breaking samples for uncertainty bands (default 100).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        dp_density : list[float]   — point estimate of DP density on grid
        dp_lower : list[float]     — 2.5th percentile band
        dp_upper : list[float]     — 97.5th percentile band
        grid : list[float]         — evaluation grid
        n_effective_components : float — inverse Herfindahl index of weights
        weights : list[float]      — top-10 stick-breaking weights
        atoms : list[float]        — top-10 atom locations
        kl_vs_parametric : float   — KL(DP || parametric best-fit)
        alpha : float              — concentration parameter used
    """
    rng = np.random.default_rng(seed)

    # Step 1: fit parametric base distribution
    all_fits = fit_all_distributions(quantiles)
    best_fit = select_best_fit(all_fits)
    grid = np.array(_make_grid(quantiles, n=200))

    # Step 2: estimate bandwidth
    sigma = _base_sigma(best_fit)
    bandwidth = _silverman_bandwidth(sigma, K_max)

    # Step 3: generate B stick-breaking samples for uncertainty
    all_densities = np.zeros((B, len(grid)))
    all_weights_list = []
    all_atoms_list = []

    for b in range(B):
        weights = _stick_breaking_weights(alpha, K_max, rng)
        atoms = _sample_atoms(best_fit, K_max, rng)
        density = _dp_density_on_grid(weights, atoms, grid, bandwidth)
        # Normalise density
        integral = np.trapezoid(density, grid)
        if integral > 1e-12:
            density /= integral
        all_densities[b] = density
        all_weights_list.append(weights)
        all_atoms_list.append(atoms)

    # Step 4: point estimate (mean across bootstrap samples)
    dp_density = np.mean(all_densities, axis=0)
    # Re-normalise
    integral = np.trapezoid(dp_density, grid)
    if integral > 1e-12:
        dp_density /= integral

    # Step 5: 95% credible bands (pointwise)
    dp_lower = np.percentile(all_densities, 2.5, axis=0)
    dp_upper = np.percentile(all_densities, 97.5, axis=0)

    # Step 6: effective number of components from first sample's weights
    ref_weights = all_weights_list[0]
    herfindahl = np.sum(ref_weights ** 2)
    n_effective = 1.0 / herfindahl if herfindahl > 1e-12 else K_max

    # Step 7: top-10 atoms and weights (from first sample, sorted by weight)
    idx_sorted = np.argsort(-ref_weights)[:10]
    top_weights = ref_weights[idx_sorted].tolist()
    top_atoms = all_atoms_list[0][idx_sorted].tolist()

    # Step 8: KL divergence from DP density to parametric best-fit
    parametric_pdf = np.array(best_fit.pdf_values)
    kl = kl_divergence(dp_density, parametric_pdf, grid)

    return {
        "dp_density": dp_density.tolist(),
        "dp_lower": dp_lower.tolist(),
        "dp_upper": dp_upper.tolist(),
        "grid": grid.tolist(),
        "n_effective_components": float(n_effective),
        "weights": top_weights,
        "atoms": top_atoms,
        "kl_vs_parametric": float(kl),
        "alpha": float(alpha),
    }
