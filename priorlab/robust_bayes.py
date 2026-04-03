"""Robust Bayesian Analysis for Prior Sensitivity.

Analyze sensitivity to prior misspecification using epsilon-contamination
classes (Berger 1984).  Given an elicited prior p_0, the contamination
class Gamma_eps = {(1-eps)*p_0 + eps*q : q arbitrary} represents all
possible priors within eps-distance of the elicited one.

This module computes:
  - Range of posterior means over Gamma_eps for multiple eps values
  - Gamma-minimax decision (robustness index for decision flipping)
  - Prior range analysis over a rectangle of Normal hyperparameters
  - Density bands (upper/lower envelope) over the contamination class
"""

import numpy as np
from scipy import stats

from priorlab.models import ElicitedQuantiles, FittedDistribution
from priorlab.fitting import fit_all_distributions, select_best_fit, _make_grid


def _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq):
    """Normal-Normal conjugate posterior.

    Prior:     theta ~ N(mu0, sigma0^2)
    Likelihood: y | theta ~ N(theta, sigma_y^2)
    Posterior:  theta | y ~ N(mu_post, var_post)
    """
    prec_prior = 1.0 / sigma0_sq
    prec_data = 1.0 / sigma_y_sq
    var_post = 1.0 / (prec_prior + prec_data)
    mu_post = var_post * (prec_prior * mu0 + prec_data * y)
    return mu_post, var_post


def _fit_to_normal_params(fitted):
    """Extract (mu, sigma) from a FittedDistribution, moment-matching if needed."""
    params = fitted.params
    family = fitted.family

    if family == "normal":
        return params["mu"], params["sigma"]
    elif family == "t":
        return params["loc"], params["scale"]
    else:
        # Moment-match from the grid
        grid = np.array(fitted.x_grid)
        pdf = np.array(fitted.pdf_values)
        integral = np.trapezoid(pdf, grid)
        if integral > 1e-12:
            pdf = pdf / integral
        mu = float(np.trapezoid(grid * pdf, grid))
        var = float(np.trapezoid((grid - mu) ** 2 * pdf, grid))
        return mu, max(np.sqrt(var), 1e-6)


def epsilon_contamination(quantiles, y, sigma_y,
                          eps_values=None, seed=42):
    """Compute posterior mean ranges under epsilon-contamination.

    For each eps, the contamination class is:
      Gamma_eps = {(1-eps)*p_0 + eps*q : q arbitrary}

    The posterior mean range is:
      Lower: (1-eps)*E_post[theta|p_0] + eps*x_min
      Upper: (1-eps)*E_post[theta|p_0] + eps*x_max

    where x_min, x_max are the support boundaries.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Elicited quantile values.
    y : float
        Observed data value.
    sigma_y : float
        Likelihood standard deviation.
    eps_values : list of float, optional
        Contamination fractions (default [0, 0.05, 0.1, 0.2, 0.3, 0.5]).
    seed : int
        Random seed (unused but kept for API consistency).

    Returns
    -------
    dict mapping eps -> {"lower": float, "upper": float}
    """
    if eps_values is None:
        eps_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    # Fit base distribution and get Normal approximation
    all_fits = fit_all_distributions(quantiles)
    best_fit = select_best_fit(all_fits)
    mu0, sigma0 = _fit_to_normal_params(best_fit)

    # Compute posterior under base prior
    sigma_y_sq = sigma_y ** 2
    sigma0_sq = sigma0 ** 2
    mu_post, _ = _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq)

    # Support boundaries (use extended grid limits)
    span = quantiles.upper - quantiles.lower
    x_min = quantiles.lower - 2.0 * span
    x_max = quantiles.upper + 2.0 * span

    result = {}
    for eps in eps_values:
        lower = (1.0 - eps) * mu_post + eps * x_min
        upper = (1.0 - eps) * mu_post + eps * x_max
        result[eps] = {"lower": float(lower), "upper": float(upper)}

    return result


def gamma_minimax(quantiles, y, sigma_y, threshold=0.0,
                  eps_values=None):
    """Find the robustness index: max eps at which the optimal decision holds.

    For a binary decision (treat if posterior mean > threshold, don't treat
    otherwise), the robustness index is the largest eps at which the decision
    remains unchanged across the entire contamination class.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    y : float
    sigma_y : float
    threshold : float
        Decision threshold (default 0.0 — treat if effect > 0).
    eps_values : list of float, optional
        Contamination fractions to scan.

    Returns
    -------
    dict with keys:
        robustness_index : float — max eps where decision is unchanged
        gamma_minimax_decision : str — "treat" or "do_not_treat"
    """
    if eps_values is None:
        eps_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                      0.35, 0.4, 0.45, 0.5]

    ranges = epsilon_contamination(quantiles, y, sigma_y, eps_values=eps_values)

    # Decision under eps=0 (no contamination)
    base_range = ranges[eps_values[0]]
    base_mean = (base_range["lower"] + base_range["upper"]) / 2.0
    base_decision = "treat" if base_mean > threshold else "do_not_treat"

    robustness_index = 0.0
    for eps in eps_values:
        rng = ranges[eps]
        # Decision is ambiguous if the range straddles the threshold
        if rng["lower"] <= threshold <= rng["upper"] and eps > 0:
            break
        robustness_index = eps

    return {
        "robustness_index": float(robustness_index),
        "gamma_minimax_decision": base_decision,
    }


def prior_range_analysis(y, sigma_y, mu_range, sigma_range):
    """Posterior sensitivity over a rectangle of Normal prior hyperparameters.

    Compute posterior mean and 95% CI at the 4 corners of
    [mu_lo, mu_hi] x [sigma_lo, sigma_hi].

    Parameters
    ----------
    y : float
        Observed data.
    sigma_y : float
        Likelihood SD.
    mu_range : tuple of (float, float)
        (mu_lo, mu_hi) for prior mean.
    sigma_range : tuple of (float, float)
        (sigma_lo, sigma_hi) for prior SD.

    Returns
    -------
    list of dict, one per corner:
        {"mu0": float, "sigma0": float, "post_mean": float, "post_ci": [lo, hi]}
    """
    sigma_y_sq = sigma_y ** 2
    results = []

    for mu0 in mu_range:
        for sigma0 in sigma_range:
            sigma0_sq = sigma0 ** 2
            mu_post, var_post = _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq)
            sd_post = np.sqrt(var_post)
            results.append({
                "mu0": float(mu0),
                "sigma0": float(sigma0),
                "post_mean": float(mu_post),
                "post_ci": [float(mu_post - 1.96 * sd_post),
                            float(mu_post + 1.96 * sd_post)],
            })

    return results


def density_bands(quantiles, y, sigma_y, eps=0.2, n_grid=200):
    """Upper and lower density envelope for the epsilon-contamination class.

    The contaminated posterior density at each grid point x is bounded by:
      lower(x) = (1-eps) * posterior_pdf(x)
      upper(x) = (1-eps) * posterior_pdf(x) + eps * peak_spike

    We normalise the upper envelope so it integrates to 1.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    y : float
    sigma_y : float
    eps : float
        Contamination fraction.
    n_grid : int
        Grid size.

    Returns
    -------
    dict with keys: density_upper (list), density_lower (list), grid (list)
    """
    all_fits = fit_all_distributions(quantiles)
    best_fit = select_best_fit(all_fits)
    mu0, sigma0 = _fit_to_normal_params(best_fit)

    # Posterior under base prior
    sigma_y_sq = sigma_y ** 2
    sigma0_sq = sigma0 ** 2
    mu_post, var_post = _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq)
    sd_post = np.sqrt(var_post)

    grid = np.linspace(mu_post - 4 * sd_post, mu_post + 4 * sd_post, n_grid)
    base_pdf = stats.norm.pdf(grid, loc=mu_post, scale=sd_post)

    # Lower envelope: scale down base posterior
    lower = (1.0 - eps) * base_pdf

    # Upper envelope: base posterior + eps * maximally concentrated contamination
    # The worst-case contamination puts all mass at each grid point,
    # so the upper envelope is (1-eps)*base + eps*delta.
    # For a smooth representation, use a narrow Gaussian spike at each point.
    # In practice, we bound pointwise: upper(x) = (1-eps)*f(x) + eps * C
    # where C is chosen so the envelope integrates to 1.
    spike_height = 1.0 / (sd_post * 0.1 * np.sqrt(2 * np.pi))
    upper = (1.0 - eps) * base_pdf + eps * spike_height * np.ones_like(grid)

    # Normalise upper envelope
    integral_upper = np.trapezoid(upper, grid)
    if integral_upper > 1e-12:
        upper = upper / integral_upper

    # Normalise lower envelope (it won't integrate to 1 by design — it's a lower bound)
    # Keep it unnormalised to show the gap

    return {
        "density_upper": upper.tolist(),
        "density_lower": lower.tolist(),
        "grid": grid.tolist(),
    }


def robust_bayesian_analysis(quantiles, y, sigma_y,
                             threshold=0.0,
                             mu_range=None, sigma_range=None,
                             eps_values=None):
    """Full robust Bayesian analysis combining all components.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    y : float
        Observed data value.
    sigma_y : float
        Likelihood SD.
    threshold : float
        Decision boundary for gamma-minimax.
    mu_range : tuple of (float, float), optional
        Prior mean range for prior_range_analysis.
        Default: (median - IQR, median + IQR).
    sigma_range : tuple of (float, float), optional
        Prior SD range.  Default: (IQR/4, IQR).
    eps_values : list of float, optional
        Contamination fractions.

    Returns
    -------
    dict with keys:
        posterior_mean_range_by_eps : dict
        robustness_index : float
        gamma_minimax_decision : str
        prior_range_results : list of dict
        density_upper : list
        density_lower : list
        is_robust : bool — decision unchanged up to eps=0.2
    """
    if eps_values is None:
        eps_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    iqr = quantiles.q3 - quantiles.q1

    if mu_range is None:
        mu_range = (quantiles.median - iqr, quantiles.median + iqr)
    if sigma_range is None:
        sigma_range = (max(iqr / 4.0, 1e-6), max(iqr, 1e-3))

    # Epsilon-contamination ranges
    eps_ranges = epsilon_contamination(quantiles, y, sigma_y,
                                       eps_values=eps_values)

    # Gamma-minimax
    gm = gamma_minimax(quantiles, y, sigma_y, threshold=threshold,
                       eps_values=eps_values)

    # Prior range analysis
    pr_results = prior_range_analysis(y, sigma_y, mu_range, sigma_range)

    # Density bands at eps=0.2
    db = density_bands(quantiles, y, sigma_y, eps=0.2)

    # Is robust? Decision unchanged at eps=0.2
    is_robust = gm["robustness_index"] >= 0.2

    return {
        "posterior_mean_range_by_eps": eps_ranges,
        "robustness_index": gm["robustness_index"],
        "gamma_minimax_decision": gm["gamma_minimax_decision"],
        "prior_range_results": pr_results,
        "density_upper": db["density_upper"],
        "density_lower": db["density_lower"],
        "is_robust": is_robust,
    }
