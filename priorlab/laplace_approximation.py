"""Laplace approximation for fast posterior inference in PriorLab.

Provides Gaussian approximation to the posterior at its mode,
with INLA-style marginals, accuracy assessment via KL divergence,
and skewness correction.
"""

import numpy as np
from scipy import stats

from priorlab.fitting import fit_normal, fit_all_distributions
from priorlab.models import ElicitedQuantiles


def _log_prior_normal(theta, mu_0, sigma_0):
    """Log-density of Normal prior."""
    return stats.norm.logpdf(theta, loc=mu_0, scale=sigma_0)


def _log_likelihood_normal(theta, y, sigma_y):
    """Log-density of Normal likelihood."""
    return stats.norm.logpdf(y, loc=theta, scale=sigma_y)


def _get_prior_dist(quantiles):
    """Fit best distribution from quantiles and return a scipy dist + family info."""
    fits = fit_all_distributions(quantiles)
    best = fits[0]  # sorted by KS distance
    family = best.family
    params = best.params

    if family == "normal":
        return stats.norm(loc=params["mu"], scale=params["sigma"]), family
    elif family == "lognormal":
        return stats.lognorm(s=params["sigma_log"], scale=np.exp(params["mu_log"])), family
    elif family == "gamma":
        return stats.gamma(a=params["shape"], scale=params["scale"]), family
    elif family == "beta":
        return stats.beta(a=params["alpha"], b=params["beta"]), family
    elif family == "halfcauchy":
        return stats.halfcauchy(scale=params["scale"]), family
    elif family == "t":
        return stats.t(df=params["df"], loc=params["loc"], scale=params["scale"]), family
    else:
        # Fallback to normal
        nf = fit_normal(quantiles)
        return stats.norm(loc=nf.params["mu"], scale=nf.params["sigma"]), "normal"


def _finite_diff_d2(func, x, h=1e-5):
    """Second derivative via central finite differences."""
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)


def _finite_diff_d3(func, x, h=1e-5):
    """Third derivative via central finite differences."""
    return (func(x + 2 * h) - 2 * func(x + h) + 2 * func(x - h) - func(x - 2 * h)) / (2 * h ** 3)


def laplace_approximation(quantiles, y, sigma_y, n_grid=200, tau2_grid_size=30):
    """Full Laplace approximation analysis.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Prior elicited quantiles.
    y : float
        Observed data point (or summary statistic).
    sigma_y : float
        Observation standard deviation.
    n_grid : int
        Grid resolution for numerical integration.
    tau2_grid_size : int
        Grid size for INLA tau2 marginal.

    Returns
    -------
    dict with keys:
        mode, laplace_sd, laplace_ci,
        inla_marginal_theta (list), inla_marginal_tau2 (list),
        kl_vs_quadrature, skewness, skew_corrected_ci,
        quadrature_mean, quadrature_sd
    """
    prior_dist, prior_family = _get_prior_dist(quantiles)

    # --- Exact Normal-Normal ---
    nfit = fit_normal(quantiles)
    mu_0 = nfit.params["mu"]
    sigma_0 = nfit.params["sigma"]

    # Exact posterior for Normal-Normal
    prec_prior = 1.0 / (sigma_0 ** 2)
    prec_data = 1.0 / (sigma_y ** 2)
    var_post = 1.0 / (prec_prior + prec_data)
    mu_post = var_post * (prec_prior * mu_0 + prec_data * y)

    # --- General Laplace (works for any fitted prior) ---
    def log_posterior(theta):
        lp = prior_dist.logpdf(theta)
        ll = stats.norm.logpdf(y, loc=theta, scale=sigma_y)
        val = lp + ll
        return val if np.isfinite(val) else -1e30

    # Grid search for mode
    span = max(abs(quantiles.upper - quantiles.lower), sigma_y) * 3
    center = mu_post  # use Normal-Normal posterior as starting guess
    grid = np.linspace(center - span, center + span, n_grid)
    log_vals = np.array([log_posterior(th) for th in grid])
    mode_idx = np.argmax(log_vals)
    mode = float(grid[mode_idx])

    # Refine mode by narrowing grid
    if 1 < mode_idx < len(grid) - 1:
        narrow_lo = grid[max(mode_idx - 2, 0)]
        narrow_hi = grid[min(mode_idx + 2, len(grid) - 1)]
        fine_grid = np.linspace(narrow_lo, narrow_hi, 100)
        fine_vals = np.array([log_posterior(th) for th in fine_grid])
        mode = float(fine_grid[np.argmax(fine_vals)])

    # Second derivative -> Laplace SD
    H = _finite_diff_d2(log_posterior, mode)
    if H >= 0:
        # Fallback: posterior is flat or concave-up at mode -> use Normal-Normal
        laplace_sd = float(np.sqrt(var_post))
    else:
        laplace_sd = float(np.sqrt(-1.0 / H))

    laplace_ci = (
        float(mode - 1.96 * laplace_sd),
        float(mode + 1.96 * laplace_sd),
    )

    # --- Quadrature (numerical integration) for ground truth ---
    quad_lo = mode - 6 * laplace_sd
    quad_hi = mode + 6 * laplace_sd
    quad_grid = np.linspace(quad_lo, quad_hi, n_grid)
    log_post_grid = np.array([log_posterior(th) for th in quad_grid])
    # Stabilize for exp
    log_post_grid_stable = log_post_grid - np.max(log_post_grid)
    post_unnorm = np.exp(log_post_grid_stable)
    Z = np.trapezoid(post_unnorm, quad_grid)
    if Z > 0:
        post_norm = post_unnorm / Z
    else:
        post_norm = np.ones_like(quad_grid) / len(quad_grid)

    quad_mean = float(np.trapezoid(quad_grid * post_norm, quad_grid))
    quad_var = float(np.trapezoid((quad_grid - quad_mean) ** 2 * post_norm, quad_grid))
    quad_sd = float(np.sqrt(max(quad_var, 0.0)))

    # --- KL divergence: Laplace vs quadrature ---
    laplace_pdf = stats.norm.pdf(quad_grid, loc=mode, scale=laplace_sd)
    laplace_pdf = np.maximum(laplace_pdf, 1e-300)
    post_safe = np.maximum(post_norm, 1e-300)
    kl_integrand = post_safe * (np.log(post_safe) - np.log(laplace_pdf))
    kl = float(np.trapezoid(kl_integrand, quad_grid))
    kl = max(kl, 0.0)

    # --- Skewness correction (third-order Laplace) ---
    d3 = _finite_diff_d3(log_posterior, mode)
    if H < 0:
        skewness = float(d3 / ((-H) ** 1.5))
    else:
        skewness = 0.0

    z_lo, z_hi = -1.96, 1.96
    q_lo_corr = mode + z_lo * laplace_sd + (z_lo ** 2 - 1) * skewness * laplace_sd / 6
    q_hi_corr = mode + z_hi * laplace_sd + (z_hi ** 2 - 1) * skewness * laplace_sd / 6
    skew_corrected_ci = (float(q_lo_corr), float(q_hi_corr))

    # --- INLA-style marginals ---
    # For 2-parameter model (theta, tau2) with tau2 = between-study variance:
    # We treat sigma_0^2 as a proxy for tau2 and grid over it.
    tau2_lo = max(sigma_0 ** 2 * 0.01, 1e-6)
    tau2_hi = sigma_0 ** 2 * 5.0
    tau2_grid = np.linspace(tau2_lo, tau2_hi, tau2_grid_size)

    inla_theta_accum = np.zeros(n_grid)
    inla_tau2_vals = []

    theta_inla_grid = np.linspace(mode - 4 * laplace_sd, mode + 4 * laplace_sd, n_grid)

    for tau2 in tau2_grid:
        # At this tau2: posterior for theta is Normal-Normal
        prec_p = 1.0 / tau2
        prec_d = 1.0 / (sigma_y ** 2)
        v = 1.0 / (prec_p + prec_d)
        m = v * (prec_p * mu_0 + prec_d * y)
        sd_inner = np.sqrt(v)

        # Marginal likelihood for this tau2
        marg_var = tau2 + sigma_y ** 2
        marg_ll = stats.norm.logpdf(y, loc=mu_0, scale=np.sqrt(marg_var))

        inla_tau2_vals.append({"tau2": float(tau2), "log_marginal": float(marg_ll)})

        # Accumulate theta marginal
        theta_dens = stats.norm.pdf(theta_inla_grid, loc=m, scale=sd_inner) * np.exp(marg_ll)
        inla_theta_accum += theta_dens

    # Normalize INLA theta marginal
    Z_theta = np.trapezoid(inla_theta_accum, theta_inla_grid)
    if Z_theta > 0:
        inla_theta_accum /= Z_theta

    inla_marginal_theta = [
        {"theta": float(th), "density": float(d)}
        for th, d in zip(theta_inla_grid, inla_theta_accum)
    ]

    # Normalize tau2 marginal
    tau2_ll = np.array([v["log_marginal"] for v in inla_tau2_vals])
    tau2_ll_stable = tau2_ll - np.max(tau2_ll)
    tau2_dens = np.exp(tau2_ll_stable)
    Z_tau2 = np.trapezoid(tau2_dens, tau2_grid)
    if Z_tau2 > 0:
        tau2_dens /= Z_tau2
    inla_marginal_tau2 = [
        {"tau2": float(t), "density": float(d)}
        for t, d in zip(tau2_grid, tau2_dens)
    ]

    return {
        "mode": mode,
        "laplace_sd": laplace_sd,
        "laplace_ci": laplace_ci,
        "inla_marginal_theta": inla_marginal_theta,
        "inla_marginal_tau2": inla_marginal_tau2,
        "kl_vs_quadrature": kl,
        "skewness": skewness,
        "skew_corrected_ci": skew_corrected_ci,
        "quadrature_mean": quad_mean,
        "quadrature_sd": quad_sd,
    }
