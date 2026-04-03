"""Prior predictive checking for PriorLab.

Simulate data from the prior to assess whether the prior is reasonable.
Includes prior predictive distribution, calibration checks,
prior-data conflict detection, and visual data for plotting.
"""

import numpy as np
from scipy import stats

from priorlab.fitting import fit_all_distributions, fit_normal
from priorlab.models import ElicitedQuantiles


def _get_prior_dist(quantiles):
    """Fit best distribution and return a scipy frozen distribution."""
    fits = fit_all_distributions(quantiles)
    best = fits[0]
    family = best.family
    params = best.params

    if family == "normal":
        return stats.norm(loc=params["mu"], scale=params["sigma"])
    elif family == "lognormal":
        return stats.lognorm(s=params["sigma_log"], scale=np.exp(params["mu_log"]))
    elif family == "gamma":
        return stats.gamma(a=params["shape"], scale=params["scale"])
    elif family == "beta":
        return stats.beta(a=params["alpha"], b=params["beta"])
    elif family == "halfcauchy":
        return stats.halfcauchy(scale=params["scale"])
    elif family == "t":
        return stats.t(df=params["df"], loc=params["loc"], scale=params["scale"])
    else:
        nf = fit_normal(quantiles)
        return stats.norm(loc=nf.params["mu"], scale=nf.params["sigma"])


def prior_predictive_check(
    quantiles, sigma_y, y_obs=None, plausible_range=None,
    n_samples=1000, seed=42, n_bins=50,
):
    """Run prior predictive checking analysis.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Prior elicited quantiles.
    sigma_y : float
        Observation noise standard deviation.
    y_obs : float or None
        Observed data point for prior predictive p-value and conflict.
    plausible_range : tuple (lo, hi) or None
        Clinically plausible range. If None, uses (-5*|median|, 5*|median|)
        with a minimum span.
    n_samples : int
        Number of prior predictive draws.
    seed : int
        Random seed for reproducibility.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict with keys:
        predictive_mean, predictive_sd, predictive_quantiles (dict),
        plausible_fraction, prior_predictive_p, conflict_index,
        is_conflicting (bool), histogram_values (list),
        histogram_bins (list), density_x (list), density_y (list)
    """
    rng = np.random.default_rng(seed)
    prior_dist = _get_prior_dist(quantiles)

    # --- Draw from prior predictive ---
    theta_samples = prior_dist.rvs(size=n_samples, random_state=rng)
    y_pred = rng.normal(loc=theta_samples, scale=sigma_y)

    # --- Summary statistics ---
    pred_mean = float(np.mean(y_pred))
    pred_sd = float(np.std(y_pred, ddof=0))

    prob_levels = {0.05: None, 0.25: None, 0.50: None, 0.75: None, 0.95: None}
    quantile_vals = np.quantile(y_pred, list(prob_levels.keys()))
    predictive_quantiles = {
        str(p): float(q) for p, q in zip(prob_levels.keys(), quantile_vals)
    }

    # --- Plausible range ---
    if plausible_range is None:
        span = max(abs(quantiles.median) * 5, 1.0)
        plausible_range = (-span, span)
    lo, hi = plausible_range
    plausible_frac = float(np.mean((y_pred >= lo) & (y_pred <= hi)))

    # --- Prior predictive p-value ---
    if y_obs is not None:
        T_obs = abs(np.mean(y_pred) - 0) if y_obs is None else abs(y_obs)
        # For each simulated dataset (single draw), compute T
        T_sim = np.abs(y_pred)
        p_pred = float(np.mean(T_sim >= T_obs))
    else:
        p_pred = float("nan")

    # --- Prior-data conflict via KL divergence ---
    if y_obs is not None:
        # Prior predictive: N(E[theta], Var[theta] + sigma_y^2) approximately
        prior_pred_mean = pred_mean
        prior_pred_var = pred_sd ** 2

        # Likelihood centered at y_obs: N(y_obs, sigma_y^2)
        lik_mean = y_obs
        lik_var = sigma_y ** 2

        # KL(prior_pred || likelihood) for two Gaussians
        kl = 0.5 * (
            np.log(lik_var / (prior_pred_var + 1e-30))
            + prior_pred_var / (lik_var + 1e-30)
            + (prior_pred_mean - lik_mean) ** 2 / (lik_var + 1e-30)
            - 1.0
        )
        kl = max(float(kl), 0.0)
        conflict_index = float(kl / (1.0 + kl))
    else:
        kl = 0.0
        conflict_index = 0.0

    is_conflicting = conflict_index > 0.5

    # --- Histogram and density for visualization ---
    hist_values, hist_edges = np.histogram(y_pred, bins=n_bins, density=True)
    histogram_values = hist_values.tolist()
    histogram_bins = hist_edges.tolist()

    # KDE density
    try:
        kde = stats.gaussian_kde(y_pred)
        density_x = np.linspace(float(np.min(y_pred)), float(np.max(y_pred)), 200)
        density_y = kde(density_x)
        density_x = density_x.tolist()
        density_y = density_y.tolist()
    except Exception:
        density_x = []
        density_y = []

    return {
        "predictive_mean": pred_mean,
        "predictive_sd": pred_sd,
        "predictive_quantiles": predictive_quantiles,
        "plausible_fraction": plausible_frac,
        "prior_predictive_p": p_pred,
        "conflict_index": conflict_index,
        "is_conflicting": is_conflicting,
        "histogram_values": histogram_values,
        "histogram_bins": histogram_bins,
        "density_x": density_x,
        "density_y": density_y,
    }
