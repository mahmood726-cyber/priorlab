"""Asymptotic theory for prior impact.

Quantifies how quickly the prior's influence vanishes with increasing data:
  - Bernstein-von Mises convergence (TV distance to BvM Normal)
  - Prior influence decay as function of n
  - Effective prior sample size
  - Posterior concentration rate (width ~ c * n^{-alpha})
  - Prior robustness horizon (n at which different priors converge)
"""

import numpy as np
from scipy import stats, optimize
from priorlab.models import ElicitedQuantiles
from priorlab.fitting import _quantile_pairs


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

def _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq, n):
    """Normal-Normal posterior after n iid observations with mean y.

    Prior:      theta ~ N(mu0, sigma0^2)
    Likelihood: y_i ~ N(theta, sigma_y^2), observe n copies with mean y
    Posterior:  N(mu_n, var_n)
    """
    prec0 = 1.0 / sigma0_sq
    prec_data = n / sigma_y_sq
    var_n = 1.0 / (prec0 + prec_data)
    mu_n = var_n * (prec0 * mu0 + prec_data * y)
    return mu_n, var_n


def _tv_distance_normals(mu1, var1, mu2, var2, n_grid=2000):
    """Total variation distance between two Normals, numerical."""
    sd1 = np.sqrt(var1)
    sd2 = np.sqrt(var2)
    lo = min(mu1 - 5 * sd1, mu2 - 5 * sd2)
    hi = max(mu1 + 5 * sd1, mu2 + 5 * sd2)
    grid = np.linspace(lo, hi, n_grid)
    p = stats.norm.pdf(grid, mu1, sd1)
    q = stats.norm.pdf(grid, mu2, sd2)
    dx = grid[1] - grid[0]
    return 0.5 * float(np.sum(np.abs(p - q)) * dx)


# ---------------------------------------------------------------------------
# 2. Bernstein-von Mises convergence
# ---------------------------------------------------------------------------

def bvm_convergence(mu0, sigma0, y, sigma_y, ns=None):
    """Check BvM theorem: posterior -> N(MLE, I^{-1}/n) as n grows.

    Parameters
    ----------
    mu0, sigma0 : prior mean and SD
    y : observed sample mean (= MLE for Normal)
    sigma_y : likelihood SD
    ns : list of sample sizes to test (default: [1,5,10,50,100,500])

    Returns
    -------
    list of {n, tv_distance} dicts
    """
    if ns is None:
        ns = [1, 5, 10, 50, 100, 500]

    sigma0_sq = sigma0 ** 2
    sigma_y_sq = sigma_y ** 2

    results = []
    for n in ns:
        # True posterior
        mu_post, var_post = _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq, n)

        # BvM approximation: N(MLE, Fisher_info^{-1}/n)
        # MLE = y, Fisher info for Normal(mean, known var) = 1/sigma_y^2
        # So BvM approx = N(y, sigma_y^2/n)
        mu_bvm = y
        var_bvm = sigma_y_sq / n

        tv = _tv_distance_normals(mu_post, var_post, mu_bvm, var_bvm)
        results.append({"n": n, "tv_distance": float(tv)})

    return results


# ---------------------------------------------------------------------------
# 3. Prior influence decay
# ---------------------------------------------------------------------------

def influence_decay(mu0, sigma0, y, sigma_y, n_max=100):
    """Ratio |posterior_mean - MLE| / |prior_mean - MLE| as function of n.

    Should decrease as O(1/n).

    Returns list of {n, ratio} dicts for n = 1..n_max.
    """
    sigma0_sq = sigma0 ** 2
    sigma_y_sq = sigma_y ** 2
    denom = abs(mu0 - y)
    if denom < 1e-12:
        # Prior mean = MLE => influence is zero everywhere
        return [{"n": n, "ratio": 0.0} for n in range(1, n_max + 1)]

    results = []
    for n in range(1, n_max + 1):
        mu_post, _ = _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq, n)
        ratio = abs(mu_post - y) / denom
        results.append({"n": n, "ratio": float(ratio)})
    return results


# ---------------------------------------------------------------------------
# 4. Effective prior sample size
# ---------------------------------------------------------------------------

def effective_prior_n(sigma0, sigma_y):
    """Effective sample size of the prior: n_0 = sigma_y^2 / sigma0^2.

    The prior has the same influence as n_0 observations.
    """
    return float(sigma_y ** 2 / sigma0 ** 2)


# ---------------------------------------------------------------------------
# 5. Posterior concentration rate
# ---------------------------------------------------------------------------

def concentration_rate(mu0, sigma0, y, sigma_y, n_max=200):
    """Fit width_n = c * n^{-alpha} to posterior 95% CI width.

    For Normal posterior, alpha should be ~0.5.

    Returns dict with alpha, c, widths (list of {n, width}).
    """
    sigma0_sq = sigma0 ** 2
    sigma_y_sq = sigma_y ** 2

    ns = list(range(1, n_max + 1))
    widths = []
    for n in ns:
        _, var_post = _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq, n)
        sd_post = np.sqrt(var_post)
        # 95% CI width = 2 * z_0.975 * sd
        width = 2 * 1.96 * sd_post
        widths.append(width)

    # Fit log(width) = log(c) - alpha * log(n) via least squares
    log_n = np.log(np.array(ns, dtype=float))
    log_w = np.log(np.array(widths))
    # Linear regression: log_w = intercept + slope * log_n
    slope, intercept = np.polyfit(log_n, log_w, 1)
    alpha = -slope
    c = np.exp(intercept)

    width_data = [{"n": n, "width": float(w)} for n, w in zip(ns, widths)]
    return {
        "concentration_alpha": float(alpha),
        "concentration_c": float(c),
        "widths": width_data,
    }


# ---------------------------------------------------------------------------
# 6. Prior robustness horizon
# ---------------------------------------------------------------------------

def robustness_horizons(experts, y, sigma_y, consensus_mu0, consensus_sigma0,
                        tv_threshold=0.05, n_max=1000):
    """Find the minimum n where each expert's posterior is TV-close to consensus.

    Parameters
    ----------
    experts : list of dicts with keys 'expert_id', 'mu0', 'sigma0'
    y : observed data
    sigma_y : likelihood SD
    consensus_mu0, consensus_sigma0 : consensus prior parameters
    tv_threshold : float, default 0.05
    n_max : int, search up to this n

    Returns
    -------
    dict expert_id -> n (minimum n for convergence)
    """
    sigma_y_sq = sigma_y ** 2
    con_sigma0_sq = consensus_sigma0 ** 2

    result = {}
    for exp in experts:
        eid = exp["expert_id"]
        mu0_e = exp["mu0"]
        sigma0_e_sq = exp["sigma0"] ** 2

        found_n = n_max  # default if never converges
        for n in range(1, n_max + 1):
            mu_e, var_e = _posterior_normal(mu0_e, sigma0_e_sq, y, sigma_y_sq, n)
            mu_c, var_c = _posterior_normal(consensus_mu0, con_sigma0_sq, y, sigma_y_sq, n)
            tv = _tv_distance_normals(mu_e, var_e, mu_c, var_c)
            if tv < tv_threshold:
                found_n = n
                break
        result[eid] = found_n

    return result


# ---------------------------------------------------------------------------
# 7. Full asymptotic analysis
# ---------------------------------------------------------------------------

def asymptotic_analysis(eq=None, y=0.0, sigma_y=1.0, experts=None):
    """Run all asymptotic analyses.

    Parameters
    ----------
    eq : ElicitedQuantiles, optional
        If provided, derive Normal prior from quantiles.
    y : float
        Observed data point.
    sigma_y : float
        Likelihood SD.
    experts : list of dicts, optional
        Each with 'expert_id', 'mu0', 'sigma0'.

    Returns
    -------
    dict with keys: bvm_convergence, influence_decay, effective_prior_n,
                    concentration_rate_alpha, concentration_c,
                    robustness_horizons.
    """
    if eq is not None:
        mu0 = eq.median
        sigma0 = (eq.q3 - eq.q1) / (2 * 0.6745)
        sigma0 = max(sigma0, 1e-6)
    else:
        mu0 = 0.0
        sigma0 = 1.0

    bvm = bvm_convergence(mu0, sigma0, y, sigma_y)
    decay = influence_decay(mu0, sigma0, y, sigma_y, n_max=100)
    eff_n = effective_prior_n(sigma0, sigma_y)
    conc = concentration_rate(mu0, sigma0, y, sigma_y, n_max=200)

    rh = {}
    if experts is not None and len(experts) > 0:
        rh = robustness_horizons(experts, y, sigma_y, mu0, sigma0)

    return {
        "bvm_convergence": bvm,
        "influence_decay": decay,
        "effective_prior_n": eff_n,
        "concentration_rate_alpha": conc["concentration_alpha"],
        "concentration_c": conc["concentration_c"],
        "robustness_horizons": rh,
    }
