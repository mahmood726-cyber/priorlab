"""Prior calibration testing.

Assess whether elicited priors are well-calibrated by checking
coverage of quantiles, PIT uniformity, and calibration-sharpness
trade-off.
"""

import numpy as np
from scipy import stats


# Nominal quantile levels matching ElicitedQuantiles structure
NOMINAL_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]


def _get_quantile_values(quantiles):
    """Extract ordered quantile values from ElicitedQuantiles."""
    return [quantiles.lower, quantiles.q1, quantiles.median, quantiles.q3, quantiles.upper]


def _make_scipy_dist(fitted):
    """Create a scipy distribution from a FittedDistribution."""
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
    elif family == "halfcauchy":
        return stats.halfcauchy(scale=p["scale"])
    elif family == "t":
        return stats.t(df=p["df"], loc=p["loc"], scale=p["scale"])
    else:
        raise ValueError(f"Unknown distribution family: {family}")


def hit_rate_test(quantiles, fitted, n_sim=1000, seed=42):
    """Check if quantile coverage matches nominal levels.

    Generate *n_sim* synthetic values from the fitted distribution and
    compute the empirical fraction falling below each elicited quantile.

    Returns
    -------
    dict mapping quantile level (float) to empirical rate (float).
    """
    rng = np.random.default_rng(seed)
    dist = _make_scipy_dist(fitted)
    samples = dist.rvs(size=n_sim, random_state=rng)

    q_values = _get_quantile_values(quantiles)
    hit_rates = {}
    for level, q_val in zip(NOMINAL_LEVELS, q_values):
        empirical = float(np.mean(samples <= q_val))
        hit_rates[level] = empirical
    return hit_rates


def pit_test(fitted, n_sim=1000, seed=42):
    """Probability integral transform test for calibration.

    If priors are well-calibrated, CDF(true_value) ~ Uniform(0,1).
    Tests uniformity via Kolmogorov-Smirnov test.

    Returns
    -------
    dict with keys: pit_values, ks_stat, p_value.
    """
    rng = np.random.default_rng(seed)
    dist = _make_scipy_dist(fitted)
    samples = dist.rvs(size=n_sim, random_state=rng)
    pit_values = dist.cdf(samples)

    ks_result = stats.kstest(pit_values, "uniform")
    return {
        "pit_values": pit_values,
        "ks_stat": float(ks_result.statistic),
        "p_value": float(ks_result.pvalue),
    }


def calibration_score(quantiles, fitted, n_sim=1000, seed=42):
    """Brier-like calibration score.

    score = mean((empirical_coverage - nominal_coverage)^2) across
    quantile levels [0.05, 0.25, 0.50, 0.75, 0.95].
    """
    hit_rates = hit_rate_test(quantiles, fitted, n_sim=n_sim, seed=seed)
    diffs_sq = [(hit_rates[level] - level) ** 2 for level in NOMINAL_LEVELS]
    return float(np.mean(diffs_sq))


def sharpness(quantiles):
    """Sharpness = width of the 90% credible interval.

    Narrower is better, but must be paired with good calibration.
    """
    return float(quantiles.upper - quantiles.lower)


def calibrate_prior(quantiles, fitted, n_sim=1000, seed=42):
    """Full calibration assessment for a single expert prior.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
    fitted : FittedDistribution
    n_sim : int
    seed : int

    Returns
    -------
    dict with keys: hit_rates, pit_ks_stat, pit_p_value,
    calibration_score, sharpness, is_calibrated.
    """
    hr = hit_rate_test(quantiles, fitted, n_sim=n_sim, seed=seed)
    pit = pit_test(fitted, n_sim=n_sim, seed=seed)
    cal_score = calibration_score(quantiles, fitted, n_sim=n_sim, seed=seed)
    sharp = sharpness(quantiles)

    return {
        "hit_rates": hr,
        "pit_ks_stat": pit["ks_stat"],
        "pit_p_value": pit["p_value"],
        "calibration_score": cal_score,
        "sharpness": sharp,
        "is_calibrated": pit["p_value"] > 0.05,
    }
