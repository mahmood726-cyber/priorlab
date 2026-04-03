"""Proper Scoring Rules for Prior Evaluation.

Decision-theoretic evaluation of elicited priors using strictly proper
scoring rules.  A scoring rule S(P, x) assigns a penalty when the
forecaster issues distribution P and outcome x materialises.  A rule
is *proper* if the expected score is optimised when the forecaster
reports their true belief.

Implemented scores:
  - Log score (logarithmic scoring rule)
  - CRPS (Continuous Ranked Probability Score)
  - Brier score (discretised)
  - Calibration-sharpness decomposition of CRPS
  - Skill score relative to Uniform reference
  - Comparative expert ranking
"""

import numpy as np
from scipy import stats

from priorlab.models import ElicitedQuantiles, FittedDistribution, ExpertPrior
from priorlab.fitting import fit_all_distributions, select_best_fit, _make_grid


def _quantile_values(eq):
    """Return the 5 elicited quantile values as a list."""
    return [eq.lower, eq.q1, eq.median, eq.q3, eq.upper]


def _get_scipy_dist(fitted):
    """Reconstruct a scipy distribution object from a FittedDistribution."""
    family = fitted.family
    p = fitted.params

    if family == "normal":
        return stats.norm(loc=p["mu"], scale=p["sigma"])
    elif family == "lognormal":
        return stats.lognorm(s=p["sigma_log"], scale=np.exp(p["mu_log"]))
    elif family == "gamma":
        return stats.gamma(a=p["shape"], scale=p["scale"])
    elif family == "beta":
        return stats.beta(a=p["alpha"], b=p["beta"])
    elif family == "t":
        return stats.t(df=p["df"], loc=p["loc"], scale=p["scale"])
    elif family == "halfcauchy":
        return stats.halfcauchy(scale=p["scale"])
    else:
        return None


def log_score(quantiles, fitted):
    """Mean log score evaluated at the 5 elicited quantile points.

    LS(p, x) = log(p(x))

    For a proper density that is < 1 at the evaluation points,
    the log score will be negative.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    fitted : FittedDistribution

    Returns
    -------
    float — mean log score (typically negative)
    """
    dist = _get_scipy_dist(fitted)
    if dist is None:
        return float("-inf")

    values = _quantile_values(quantiles)
    scores = []
    for x in values:
        pdf_val = dist.pdf(x)
        if pdf_val > 1e-300:
            scores.append(np.log(pdf_val))
        else:
            scores.append(-700.0)  # log(1e-300) ~ -690

    return float(np.mean(scores))


def crps(quantiles, fitted, n_grid=500):
    """Continuous Ranked Probability Score, averaged over quantile points.

    CRPS(F, x) = integral (F(t) - I(t >= x))^2 dt

    Lower is better.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    fitted : FittedDistribution
    n_grid : int
        Grid resolution for numerical integration.

    Returns
    -------
    float — mean CRPS across the 5 quantile points
    """
    dist = _get_scipy_dist(fitted)
    if dist is None:
        return float("inf")

    values = _quantile_values(quantiles)
    grid = np.array(_make_grid(quantiles, n=n_grid))
    dx = grid[1] - grid[0]
    cdf_grid = dist.cdf(grid)

    total_crps = 0.0
    for x in values:
        indicator = (grid >= x).astype(float)
        integrand = (cdf_grid - indicator) ** 2
        total_crps += np.sum(integrand) * dx

    return float(total_crps / len(values))


def brier_score(quantiles, fitted, n_bins=20):
    """Discretised Brier score.

    Bin the support, compute predicted probability per bin from the fitted
    distribution, and compare against an empirical distribution formed by
    placing the 5 quantile observations into bins.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    fitted : FittedDistribution
    n_bins : int
        Number of bins (default 20).

    Returns
    -------
    float — mean squared error across bins
    """
    dist = _get_scipy_dist(fitted)
    if dist is None:
        return float("inf")

    values = _quantile_values(quantiles)
    span = quantiles.upper - quantiles.lower
    lo = quantiles.lower - 0.2 * span
    hi = quantiles.upper + 0.2 * span
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    # Predicted: probability mass in each bin from fitted CDF
    p_pred = np.diff(dist.cdf(bin_edges))

    # Observed: histogram of the 5 quantile values, normalised
    p_obs, _ = np.histogram(values, bins=bin_edges, density=False)
    p_obs = p_obs.astype(float)
    total = p_obs.sum()
    if total > 0:
        p_obs /= total

    # Brier score: mean squared difference
    bs = float(np.mean((p_pred - p_obs) ** 2))
    return bs


def crps_decomposition(quantiles, fitted, n_grid=500):
    """Calibration-sharpness decomposition of CRPS.

    CRPS = Reliability - Resolution + Uncertainty

    - Reliability: how well CDF probabilities match empirical frequencies
    - Resolution: discrimination ability
    - Uncertainty: inherent variability of observations

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    fitted : FittedDistribution
    n_grid : int

    Returns
    -------
    dict with keys: reliability, resolution, crps_total
    """
    dist = _get_scipy_dist(fitted)
    if dist is None:
        return {"reliability": float("inf"), "resolution": 0.0,
                "crps_total": float("inf")}

    values = np.array(_quantile_values(quantiles))
    n = len(values)

    # CDF values at the quantile points (PIT values)
    pit_values = dist.cdf(values)

    # Sort PIT values
    pit_sorted = np.sort(pit_values)

    # Expected uniform quantiles
    expected = (np.arange(1, n + 1) - 0.5) / n

    # Reliability: mean squared deviation of PIT from uniform
    reliability = float(np.mean((pit_sorted - expected) ** 2))

    # Resolution: variance of PIT values (higher = better discrimination)
    resolution = float(np.var(pit_sorted))

    # Total CRPS from the main function
    crps_total = crps(quantiles, fitted, n_grid)

    return {
        "reliability": reliability,
        "resolution": resolution,
        "crps_total": crps_total,
    }


def skill_score(quantiles, fitted, n_grid=500):
    """Skill score relative to a Uniform reference distribution.

    SS = 1 - CRPS_prior / CRPS_reference

    SS > 0: prior is better than uniform.
    SS = 1: perfect prior.
    SS < 0: prior is worse than uniform (unusual).

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    fitted : FittedDistribution
    n_grid : int

    Returns
    -------
    float — skill score in (-inf, 1]
    """
    crps_prior = crps(quantiles, fitted, n_grid)

    # Reference: Uniform over the extended support
    span = quantiles.upper - quantiles.lower
    lo = quantiles.lower - 0.2 * span
    hi = quantiles.upper + 0.2 * span
    ref_dist = stats.uniform(loc=lo, scale=hi - lo)

    # Build a FittedDistribution-like object for the uniform
    # We compute CRPS directly using the reference distribution
    values = _quantile_values(quantiles)
    grid = np.array(_make_grid(quantiles, n=n_grid))
    dx = grid[1] - grid[0]
    cdf_ref = ref_dist.cdf(grid)

    crps_ref = 0.0
    for x in values:
        indicator = (grid >= x).astype(float)
        integrand = (cdf_ref - indicator) ** 2
        crps_ref += np.sum(integrand) * dx
    crps_ref /= len(values)

    if crps_ref < 1e-12:
        return 0.0

    ss = 1.0 - crps_prior / crps_ref
    return float(ss)


def comparative_scoring(experts, n_grid=500):
    """Score and rank multiple experts' priors.

    For each expert with fitted distributions, compute log score, CRPS,
    Brier score, and a total (negative CRPS, since lower CRPS is better,
    total = -CRPS so higher is better).

    Parameters
    ----------
    experts : list of ExpertPrior
        Each must have .quantiles and .best_fit set.
    n_grid : int

    Returns
    -------
    list of dict:
        {"expert_id": str, "log_score": float, "crps": float,
         "brier_score": float, "total_score": float}
    sorted by total_score descending (best first).
    """
    rankings = []
    for expert in experts:
        if expert.quantiles is None or expert.best_fit is None:
            continue
        ls = log_score(expert.quantiles, expert.best_fit)
        cr = crps(expert.quantiles, expert.best_fit, n_grid)
        bs = brier_score(expert.quantiles, expert.best_fit)
        # Total: higher is better.  Use negative CRPS (lower CRPS = better).
        total = -cr
        rankings.append({
            "expert_id": expert.expert_id,
            "log_score": ls,
            "crps": cr,
            "brier_score": bs,
            "total_score": total,
        })

    rankings.sort(key=lambda r: r["total_score"], reverse=True)
    return rankings


def score_prior(quantiles, fitted=None, experts=None, n_grid=500):
    """Full scoring analysis for an elicited prior.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        The elicited quantile values (used if fitted is None to auto-fit).
    fitted : FittedDistribution, optional
        Pre-fitted distribution.  If None, auto-fits from quantiles.
    experts : list of ExpertPrior, optional
        For comparative ranking.
    n_grid : int

    Returns
    -------
    dict with keys:
        log_score : float
        crps : float
        brier_score : float
        reliability : float
        resolution : float
        skill_score : float
        expert_rankings : list of dict (if experts provided)
        best_expert : str or None
    """
    if fitted is None:
        all_fits = fit_all_distributions(quantiles)
        fitted = select_best_fit(all_fits)

    ls = log_score(quantiles, fitted)
    cr = crps(quantiles, fitted, n_grid)
    bs = brier_score(quantiles, fitted)
    decomp = crps_decomposition(quantiles, fitted, n_grid)
    ss = skill_score(quantiles, fitted, n_grid)

    result = {
        "log_score": ls,
        "crps": cr,
        "brier_score": bs,
        "reliability": decomp["reliability"],
        "resolution": decomp["resolution"],
        "skill_score": ss,
        "expert_rankings": [],
        "best_expert": None,
    }

    if experts is not None and len(experts) > 0:
        rankings = comparative_scoring(experts, n_grid)
        result["expert_rankings"] = rankings
        if rankings:
            result["best_expert"] = rankings[0]["expert_id"]

    return result
