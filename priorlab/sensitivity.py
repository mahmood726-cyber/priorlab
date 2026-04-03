"""Prior sensitivity analysis.

Quantify how sensitive the posterior is to the choice of prior,
using local derivatives, global sweeps, and leave-one-expert-out
influence analysis.  All computations assume a Normal-Normal
conjugate model unless stated otherwise.
"""

import numpy as np


def _posterior_normal(mu0, sigma0_sq, y, sigma_y_sq):
    """Posterior mean and variance for Normal prior + Normal likelihood.

    Prior:  theta ~ N(mu0, sigma0^2)
    Data:   y | theta ~ N(theta, sigma_y^2)
    Posterior: theta | y ~ N(mu_post, var_post)
    """
    precision_prior = 1.0 / sigma0_sq
    precision_data = 1.0 / sigma_y_sq
    var_post = 1.0 / (precision_prior + precision_data)
    mu_post = var_post * (precision_prior * mu0 + precision_data * y)
    return mu_post, var_post


def local_sensitivity(mu0, sigma0, y, sigma_y):
    """Analytic derivatives of posterior mean w.r.t. prior hyperparameters.

    Parameters
    ----------
    mu0 : float
        Prior mean.
    sigma0 : float
        Prior standard deviation (>0).
    y : float
        Observed data (likelihood mean).
    sigma_y : float
        Likelihood standard deviation (>0).

    Returns
    -------
    dict with keys:
        d_mu_post_d_mu0 : shrinkage factor sigma_y^2/(sigma0^2+sigma_y^2)
        d_mu_post_d_sigma0_sq : (y-mu0)*sigma_y^2/(sigma0^2+sigma_y^2)^2
    """
    s0sq = sigma0 ** 2
    sysq = sigma_y ** 2
    denom = s0sq + sysq
    d_mu0 = sysq / denom            # shrinkage factor — in (0, 1)
    d_sigma0_sq = (y - mu0) * sysq / (denom ** 2)
    return {
        "d_mu_post_d_mu0": float(d_mu0),
        "d_mu_post_d_sigma0_sq": float(d_sigma0_sq),
    }


def global_sensitivity(mu0, sigma0, y, sigma_y, n_steps=20):
    """Sweep prior hyperparameters and record posterior for each.

    Vary mu0 over [mu0 - 2*sigma0, mu0 + 2*sigma0] and
    sigma0 over [sigma0*0.5, sigma0*2.0] in *n_steps* each.

    Returns
    -------
    dict with keys:
        mu_sweep : list of {mu0, post_mean, post_ci_lo, post_ci_hi}
        sigma_sweep : list of {sigma0, post_mean, post_ci_lo, post_ci_hi}
        robustness_ratio : float  (max / min posterior mean across mu sweep)
    """
    sysq = sigma_y ** 2

    # --- mu sweep ---
    mu_lo = mu0 - 2 * sigma0
    mu_hi = mu0 + 2 * sigma0
    mu_vals = np.linspace(mu_lo, mu_hi, n_steps)
    mu_sweep = []
    post_means_mu = []
    for m in mu_vals:
        pm, pv = _posterior_normal(m, sigma0 ** 2, y, sysq)
        ps = np.sqrt(pv)
        mu_sweep.append({
            "mu0": float(m),
            "post_mean": float(pm),
            "post_ci_lo": float(pm - 1.96 * ps),
            "post_ci_hi": float(pm + 1.96 * ps),
        })
        post_means_mu.append(pm)

    # --- sigma sweep ---
    sig_lo = sigma0 * 0.5
    sig_hi = sigma0 * 2.0
    sig_vals = np.linspace(sig_lo, sig_hi, n_steps)
    sigma_sweep = []
    for s in sig_vals:
        pm, pv = _posterior_normal(mu0, s ** 2, y, sysq)
        ps = np.sqrt(pv)
        sigma_sweep.append({
            "sigma0": float(s),
            "post_mean": float(pm),
            "post_ci_lo": float(pm - 1.96 * ps),
            "post_ci_hi": float(pm + 1.96 * ps),
        })

    # --- robustness ratio ---
    arr = np.array(post_means_mu)
    mn = np.min(np.abs(arr))
    mx = np.max(np.abs(arr))
    if mn > 1e-12:
        robustness_ratio = float(mx / mn)
    else:
        robustness_ratio = float("inf")

    return {
        "mu_sweep": mu_sweep,
        "sigma_sweep": sigma_sweep,
        "robustness_ratio": robustness_ratio,
    }


def expert_influence(expert_priors, y, sigma_y):
    """Leave-one-expert-out influence on the aggregated posterior.

    For each expert, remove them and recompute the aggregated prior
    (equal-weight linear pool of remaining Normal priors).  Then
    compute the posterior with the aggregated prior and measure the
    change compared to the full-panel posterior.

    Parameters
    ----------
    expert_priors : list of dict
        Each dict has keys 'mu0' and 'sigma0' (the expert's Normal prior).
    y : float
        Observed data value.
    sigma_y : float
        Data standard deviation.

    Returns
    -------
    dict  expert_id (int) -> delta_posterior_mean (float)
    """
    n = len(expert_priors)
    if n < 2:
        return {}

    sysq = sigma_y ** 2

    # Full-panel aggregated prior: equal-weight Normal mixture
    # Approximate by moment-matching
    def _moment_match(priors):
        k = len(priors)
        mus = [p["mu0"] for p in priors]
        sigs = [p["sigma0"] for p in priors]
        agg_mu = np.mean(mus)
        agg_var = np.mean([s ** 2 + m ** 2 for s, m in zip(sigs, mus)]) - agg_mu ** 2
        agg_var = max(agg_var, 1e-12)
        return agg_mu, agg_var

    full_mu, full_var = _moment_match(expert_priors)
    full_post, _ = _posterior_normal(full_mu, full_var, y, sysq)

    influence = {}
    for i in range(n):
        reduced = [p for j, p in enumerate(expert_priors) if j != i]
        red_mu, red_var = _moment_match(reduced)
        red_post, _ = _posterior_normal(red_mu, red_var, y, sysq)
        influence[i] = float(red_post - full_post)

    return influence


def sensitivity_analysis(mu0, sigma0, y, sigma_y, expert_priors=None, n_steps=20):
    """Full sensitivity analysis combining local, global, and influence.

    Parameters
    ----------
    mu0 : float
        Prior mean (for local/global analysis).
    sigma0 : float
        Prior SD (for local/global analysis).
    y : float
        Observed data.
    sigma_y : float
        Data SD.
    expert_priors : list of dict, optional
        For influence analysis.  Each dict: {'mu0': float, 'sigma0': float}.
    n_steps : int
        Number of grid points for global sweep.

    Returns
    -------
    dict with keys: local_sensitivity, global_mu_sweep, global_sigma_sweep,
    expert_influence, robustness_ratio.
    """
    loc = local_sensitivity(mu0, sigma0, y, sigma_y)
    glob = global_sensitivity(mu0, sigma0, y, sigma_y, n_steps=n_steps)

    infl = {}
    if expert_priors is not None and len(expert_priors) >= 2:
        infl = expert_influence(expert_priors, y, sigma_y)

    return {
        "local_sensitivity": loc,
        "global_mu_sweep": glob["mu_sweep"],
        "global_sigma_sweep": glob["sigma_sweep"],
        "expert_influence": infl,
        "robustness_ratio": glob["robustness_ratio"],
    }
