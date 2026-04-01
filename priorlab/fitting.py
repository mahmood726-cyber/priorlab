import numpy as np
from scipy import stats
from priorlab.models import FittedDistribution, ElicitedQuantiles


def _quantile_pairs(eq):
    """Return list of (probability, quantile_value) pairs."""
    return [
        (eq.lower_p, eq.lower),
        (0.25, eq.q1),
        (0.50, eq.median),
        (0.75, eq.q3),
        (eq.upper_p, eq.upper),
    ]


def _ks_distance(dist, eq):
    """KS distance: max|F(q_i) - p_i| at elicited quantiles."""
    pairs = _quantile_pairs(eq)
    return max(abs(dist.cdf(q) - p) for p, q in pairs)


def _make_grid(eq, n=200):
    span = eq.upper - eq.lower
    lo = eq.lower - 0.2 * span
    hi = eq.upper + 0.2 * span
    return np.linspace(lo, hi, n)


def _build_result(family, params, dist, eq):
    grid = _make_grid(eq)
    pdf = dist.pdf(grid)
    ks = _ks_distance(dist, eq)
    pairs = _quantile_pairs(eq)
    cdf_at = {f"p{p:.2f}": float(dist.cdf(q)) for p, q in pairs}
    loglik = sum(dist.logpdf(q) for _, q in pairs if np.isfinite(dist.logpdf(q)))
    k = len(params)
    aic = -2 * loglik + 2 * k if np.isfinite(loglik) else 1e6
    return FittedDistribution(
        family=family, params=params, ks_distance=round(ks, 6),
        aic=round(aic, 2),
        x_grid=grid.tolist(), pdf_values=pdf.tolist(), cdf_at_quantiles=cdf_at,
    )


def fit_normal(eq):
    mu = eq.median
    sigma = (eq.q3 - eq.q1) / (2 * 0.6745)
    sigma = max(sigma, 1e-6)
    dist = stats.norm(loc=mu, scale=sigma)
    return _build_result("normal", {"mu": round(mu, 6), "sigma": round(sigma, 6)}, dist, eq)


def fit_lognormal(eq):
    if eq.median <= 0:
        return FittedDistribution(family="lognormal", ks_distance=999.0)
    mu_log = np.log(eq.median)
    if eq.q3 > 0 and eq.q1 > 0:
        sigma_log = (np.log(eq.q3) - np.log(eq.q1)) / (2 * 0.6745)
    else:
        sigma_log = 0.5
    sigma_log = max(sigma_log, 1e-6)
    dist = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))
    return _build_result("lognormal",
                         {"mu_log": round(mu_log, 6), "sigma_log": round(sigma_log, 6)},
                         dist, eq)


def fit_gamma(eq):
    if eq.median <= 0:
        return FittedDistribution(family="gamma", ks_distance=999.0)
    mean_est = eq.median
    var_est = ((eq.q3 - eq.q1) / 1.349) ** 2
    var_est = max(var_est, 1e-6)
    shape = (mean_est ** 2) / var_est
    scale = var_est / mean_est
    shape = max(shape, 0.1)
    scale = max(scale, 1e-6)
    dist = stats.gamma(a=shape, scale=scale)
    return _build_result("gamma",
                         {"shape": round(shape, 6), "scale": round(scale, 6)},
                         dist, eq)


def fit_beta(eq):
    if eq.lower < 0 or eq.upper > 1:
        return FittedDistribution(family="beta", ks_distance=999.0)
    mean_est = eq.median
    var_est = ((eq.q3 - eq.q1) / 1.349) ** 2
    var_est = min(var_est, mean_est * (1 - mean_est) * 0.99)
    var_est = max(var_est, 1e-6)
    common = mean_est * (1 - mean_est) / var_est - 1
    common = max(common, 0.1)
    alpha = mean_est * common
    beta_p = (1 - mean_est) * common
    alpha = max(alpha, 0.1)
    beta_p = max(beta_p, 0.1)
    dist = stats.beta(a=alpha, b=beta_p)
    return _build_result("beta",
                         {"alpha": round(alpha, 6), "beta": round(beta_p, 6)},
                         dist, eq)


def fit_halfcauchy(eq):
    if eq.median <= 0:
        return FittedDistribution(family="halfcauchy", ks_distance=999.0)
    scale = eq.median
    dist = stats.halfcauchy(scale=scale)
    return _build_result("halfcauchy", {"scale": round(scale, 6)}, dist, eq)


def fit_t(eq):
    sigma = (eq.q3 - eq.q1) / (2 * 0.6745)
    sigma = max(sigma, 1e-6)
    best = None
    for df in [1, 2, 3, 5, 10, 20, 30]:
        scale = sigma * np.sqrt(df / (df - 2)) if df > 2 else sigma
        dist = stats.t(df=df, loc=eq.median, scale=scale)
        ks = _ks_distance(dist, eq)
        if best is None or ks < best[0]:
            best = (ks, df, scale, dist)
    _, df, scale, dist = best
    return _build_result("t",
                         {"df": df, "loc": round(eq.median, 6), "scale": round(scale, 6)},
                         dist, eq)


def fit_all_distributions(eq):
    fits = [fit_normal(eq), fit_gamma(eq), fit_halfcauchy(eq), fit_t(eq)]
    ln = fit_lognormal(eq)
    if ln.ks_distance < 900:
        fits.append(ln)
    bt = fit_beta(eq)
    if bt.ks_distance < 900:
        fits.append(bt)
    return sorted(fits, key=lambda f: f.ks_distance)


def select_best_fit(fits):
    return min(fits, key=lambda f: f.ks_distance)
