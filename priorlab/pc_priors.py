"""Penalized complexity (PC) priors for heterogeneity parameters.

Based on Simpson et al. (2017): the base model is tau=0 (no heterogeneity).
The PC prior for tau is exponential, with lambda calibrated from a user
statement P(tau > U) = alpha.
"""

import numpy as np
from scipy.stats import halfcauchy as _halfcauchy


def compute_lambda(upper_bound, alpha):
    """Derive lambda from P(tau > U) = alpha.

    lambda = -log(alpha) / U
    """
    if upper_bound <= 0:
        raise ValueError("upper_bound must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    return -np.log(alpha) / upper_bound


def pc_prior_tau(tau_grid, lam):
    """PC prior density on the SD scale: p(tau) = lambda * exp(-lambda * tau), tau > 0."""
    tau = np.asarray(tau_grid, dtype=float)
    density = np.where(tau > 0, lam * np.exp(-lam * tau), 0.0)
    return density


def pc_prior_tau2(tau2_grid, lam):
    """PC prior density on the variance scale via Jacobian.

    p(tau^2) = lambda / (2 * sqrt(tau^2)) * exp(-lambda * sqrt(tau^2))
    """
    tau2 = np.asarray(tau2_grid, dtype=float)
    sqrt_tau2 = np.sqrt(np.maximum(tau2, 0.0))
    density = np.where(
        tau2 > 0,
        lam / (2.0 * sqrt_tau2) * np.exp(-lam * sqrt_tau2),
        0.0,
    )
    return density


def pc_prior_log_tau(log_tau_grid, lam):
    """PC prior density on the log(tau) scale via Jacobian.

    If eta = log(tau), then tau = exp(eta) and
    p(eta) = lambda * exp(eta) * exp(-lambda * exp(eta)).
    """
    eta = np.asarray(log_tau_grid, dtype=float)
    tau = np.exp(eta)
    density = lam * tau * np.exp(-lam * tau)
    return density


def halfcauchy_density(tau_grid, scale):
    """Half-Cauchy(0, scale) density for comparison."""
    tau = np.asarray(tau_grid, dtype=float)
    return _halfcauchy(scale=scale).pdf(tau)


def kl_divergence_on_grid(p, q, grid, epsilon=1e-300):
    """KL(p||q) computed on a discrete grid.  Shared helper."""
    p = np.maximum(np.asarray(p, dtype=float), epsilon)
    q = np.maximum(np.asarray(q, dtype=float), epsilon)
    integrand = p * np.log(p / q)
    return float(np.trapezoid(integrand, grid))


def compute_pc_priors(upper_bound, alpha, halfcauchy_scale=None,
                      n_grid=500, tau_max=None):
    """Compute PC priors on all three parameterizations plus half-Cauchy comparison.

    Parameters
    ----------
    upper_bound : float
        U such that P(tau > U) = alpha.
    alpha : float
        Tail probability (e.g. 0.05).
    halfcauchy_scale : float or None
        Scale for the comparison half-Cauchy.  Defaults to upper_bound.
    n_grid : int
        Number of grid points.
    tau_max : float or None
        Upper limit of grid.  Defaults to 3 * upper_bound.

    Returns
    -------
    dict with keys: tau_grid, pc_tau_density, pc_tau2_density,
    halfcauchy_density, lambda_param, kl_pc_vs_halfcauchy.
    """
    lam = compute_lambda(upper_bound, alpha)

    if tau_max is None:
        tau_max = max(3.0 * upper_bound, 1.0)
    if halfcauchy_scale is None:
        halfcauchy_scale = upper_bound

    tau_grid = np.linspace(1e-6, tau_max, n_grid)

    pc_tau = pc_prior_tau(tau_grid, lam)
    pc_tau2_grid = tau_grid ** 2
    pc_tau2 = pc_prior_tau2(pc_tau2_grid, lam)
    hc = halfcauchy_density(tau_grid, halfcauchy_scale)

    # KL(PC || HalfCauchy) on the tau-scale grid
    kl = kl_divergence_on_grid(pc_tau, hc, tau_grid)

    return {
        "tau_grid": tau_grid,
        "pc_tau_density": pc_tau,
        "pc_tau2_density": pc_tau2,
        "halfcauchy_density": hc,
        "lambda_param": float(lam),
        "kl_pc_vs_halfcauchy": kl,
    }
