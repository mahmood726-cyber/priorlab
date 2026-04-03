"""Generalized Method of Moments for prior fitting.

Alternative to quantile matching: estimate raw moments from the 5 elicited
quantiles via trapezoidal quadrature on the empirical quantile function, then
fit each distributional family by matching theoretical moments to empirical
ones.  Includes Hansen's J-test for overidentification and KL-based
comparison between GMM and quantile-matching fits.
"""

import numpy as np
from scipy import optimize, stats
from priorlab.models import ElicitedQuantiles
from priorlab.fitting import fit_all_distributions, _quantile_pairs


# ---------------------------------------------------------------------------
# 1. Raw moments from quantiles
# ---------------------------------------------------------------------------

def _raw_moments_from_quantiles(eq, max_k=4):
    """Estimate E[X], E[X^2], E[X^3], E[X^4] via trapezoidal quadrature.

    We treat the five quantiles as an empirical quantile function Q(p) at
    probabilities p_i, then compute E[X^k] = integral_0^1 Q(p)^k dp using
    the trapezoidal rule on those points.
    """
    pairs = _quantile_pairs(eq)  # [(p, q), ...]
    # Add boundary pseudo-points at p=0 and p=1 by extrapolation
    # Use linear extrapolation from the two nearest points
    p_vals = [p for p, _ in pairs]
    q_vals = [q for _, q in pairs]

    # Prepend p=0 point (extrapolate from first two)
    if p_vals[0] > 0:
        slope = (q_vals[1] - q_vals[0]) / (p_vals[1] - p_vals[0])
        q0 = q_vals[0] - slope * p_vals[0]
        p_vals = [0.0] + p_vals
        q_vals = [q0] + q_vals

    # Append p=1 point (extrapolate from last two)
    if p_vals[-1] < 1:
        slope = (q_vals[-1] - q_vals[-2]) / (p_vals[-1] - p_vals[-2])
        q1 = q_vals[-1] + slope * (1.0 - p_vals[-1])
        p_vals = p_vals + [1.0]
        q_vals = q_vals + [q1]

    p_arr = np.array(p_vals)
    q_arr = np.array(q_vals)

    moments = []
    for k in range(1, max_k + 1):
        integrand = q_arr ** k
        mk = float(np.trapezoid(integrand, p_arr))
        moments.append(mk)
    return moments


# ---------------------------------------------------------------------------
# 2. Theoretical moment functions per family
# ---------------------------------------------------------------------------

def _normal_moments(mu, sigma):
    """Raw moments of N(mu, sigma^2)."""
    m1 = mu
    m2 = mu ** 2 + sigma ** 2
    m3 = mu ** 3 + 3 * mu * sigma ** 2
    m4 = mu ** 4 + 6 * mu ** 2 * sigma ** 2 + 3 * sigma ** 4
    return [m1, m2, m3, m4]


def _lognormal_moments(mu_ln, sigma_ln):
    """Raw moments of LogNormal(mu_ln, sigma_ln^2)."""
    moments = []
    for k in range(1, 5):
        moments.append(float(np.exp(k * mu_ln + 0.5 * k ** 2 * sigma_ln ** 2)))
    return moments


def _gamma_moments(alpha, beta):
    """Raw moments of Gamma(alpha, scale=1/beta) parameterised as shape, rate."""
    # E[X^k] = product_{i=0}^{k-1} (alpha + i) / beta^k
    moments = []
    for k in range(1, 5):
        prod = 1.0
        for i in range(k):
            prod *= (alpha + i)
        moments.append(prod / beta ** k)
    return moments


# ---------------------------------------------------------------------------
# 3. Method-of-Moments (closed form) per family
# ---------------------------------------------------------------------------

def _mom_normal(m):
    """Normal MoM: mu = m1, sigma^2 = m2 - m1^2."""
    mu = m[0]
    var = m[1] - m[0] ** 2
    var = max(var, 1e-12)
    return {"mu": mu, "sigma": np.sqrt(var)}


def _mom_lognormal(m):
    """LogNormal MoM: from m1, m2."""
    m1, m2 = m[0], m[1]
    if m1 <= 0 or m2 <= 0:
        return None
    var_ratio = m2 / m1 ** 2
    if var_ratio <= 0:
        return None
    sigma_ln = np.sqrt(np.log(var_ratio))
    mu_ln = np.log(m1) - 0.5 * sigma_ln ** 2
    sigma_ln = max(sigma_ln, 1e-12)
    return {"mu_ln": mu_ln, "sigma_ln": sigma_ln}


def _mom_gamma(m):
    """Gamma MoM: alpha = m1^2/(m2-m1^2), rate = m1/(m2-m1^2)."""
    m1, m2 = m[0], m[1]
    var = m2 - m1 ** 2
    if var <= 0 or m1 <= 0:
        return None
    alpha = m1 ** 2 / var
    beta = m1 / var  # rate
    alpha = max(alpha, 1e-6)
    beta = max(beta, 1e-6)
    return {"alpha": alpha, "beta": beta}


# ---------------------------------------------------------------------------
# 4. GMM objective + optimisation
# ---------------------------------------------------------------------------

def _gmm_objective(params, moment_fn, empirical_moments, weights):
    """Weighted sum of squared moment deviations."""
    try:
        theoretical = moment_fn(*params)
    except (ValueError, OverflowError, ZeroDivisionError):
        return 1e12
    residuals = np.array(theoretical) - np.array(empirical_moments)
    return float(np.sum(weights * residuals ** 2))


def _optimal_weights(empirical_moments):
    """Inverse-variance weighting: w_k = 1 / (m_k^2 + 1) as heuristic."""
    return np.array([1.0 / (mk ** 2 + 1.0) for mk in empirical_moments])


def _gmm_fit_family(family, moment_fn, x0, bounds, empirical_moments, n_params):
    """Run L-BFGS-B minimisation for a single family."""
    weights = _optimal_weights(empirical_moments)
    try:
        res = optimize.minimize(
            _gmm_objective, x0, args=(moment_fn, empirical_moments, weights),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-14},
        )
        obj = res.fun
        params_opt = res.x
    except Exception:
        return None

    # Hansen's J-test: J = n * obj ~ chi2(n_moments - n_params)
    n_moments = len(empirical_moments)
    df_j = n_moments - n_params
    j_stat = float(obj * 5)  # pseudo sample-size = 5 quantiles
    j_pvalue = float(1.0 - stats.chi2.cdf(j_stat, max(df_j, 1))) if df_j > 0 else 1.0

    return {
        "family": family,
        "params": params_opt.tolist(),
        "j_stat": j_stat,
        "j_pvalue": j_pvalue,
        "objective": float(obj),
    }


# ---------------------------------------------------------------------------
# 5. KL divergence (numerical, on a grid)
# ---------------------------------------------------------------------------

def _kl_on_grid(p_pdf, q_pdf, grid):
    """KL(p || q) numerically on a shared grid."""
    eps = 1e-300
    p = np.maximum(np.asarray(p_pdf), eps)
    q = np.maximum(np.asarray(q_pdf), eps)
    integrand = p * np.log(p / q)
    return float(np.trapezoid(integrand, grid))


def _empirical_pdf_from_quantiles(eq, grid):
    """Build a rough empirical PDF from quantile information using a KDE-like approach."""
    pairs = _quantile_pairs(eq)
    q_vals = np.array([q for _, q in pairs])
    # Use a simple Gaussian KDE with bandwidth = IQR/4
    iqr = eq.q3 - eq.q1
    bw = max(iqr / 4.0, 1e-6)
    pdf = np.zeros_like(grid, dtype=float)
    for qv in q_vals:
        pdf += stats.norm.pdf(grid, loc=qv, scale=bw)
    pdf /= len(q_vals)
    # Normalise
    integral = np.trapezoid(pdf, grid)
    if integral > 0:
        pdf /= integral
    return pdf


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def fit_gmm(eq):
    """Fit all families via Generalized Method of Moments.

    Parameters
    ----------
    eq : ElicitedQuantiles

    Returns
    -------
    dict with keys:
        moment_estimates : list of [m1, m2, m3, m4]
        gmm_fits : dict family -> {params, j_stat, j_pvalue, objective}
        best_gmm_family : str
        kl_gmm_vs_quantile : dict family -> {gmm_kl, qm_kl}
        overall_best_method : str  ("gmm" or "quantile_matching")
    """
    moments = _raw_moments_from_quantiles(eq)
    m = moments  # shorthand

    gmm_fits = {}

    # Normal: 2 params (mu, sigma)
    x0_n = [m[0], max(np.sqrt(max(m[1] - m[0] ** 2, 1e-6)), 1e-3)]
    res_n = _gmm_fit_family(
        "normal", _normal_moments, x0_n,
        [(-np.inf, np.inf), (1e-6, np.inf)], m, 2,
    )
    if res_n is not None:
        gmm_fits["normal"] = res_n

    # LogNormal: 2 params (mu_ln, sigma_ln)
    if m[0] > 0:
        mom_ln = _mom_lognormal(m)
        if mom_ln is not None:
            x0_ln = [mom_ln["mu_ln"], mom_ln["sigma_ln"]]
        else:
            x0_ln = [0.0, 1.0]
        res_ln = _gmm_fit_family(
            "lognormal", _lognormal_moments, x0_ln,
            [(-np.inf, np.inf), (1e-6, 10.0)], m, 2,
        )
        if res_ln is not None:
            gmm_fits["lognormal"] = res_ln

    # Gamma: 2 params (alpha, rate=beta)
    if m[0] > 0:
        mom_g = _mom_gamma(m)
        if mom_g is not None:
            x0_g = [mom_g["alpha"], mom_g["beta"]]
        else:
            x0_g = [2.0, 2.0]
        res_g = _gmm_fit_family(
            "gamma", _gamma_moments, x0_g,
            [(1e-6, 1e4), (1e-6, 1e4)], m, 2,
        )
        if res_g is not None:
            gmm_fits["gamma"] = res_g

    # Find best GMM family
    best_family = None
    best_obj = np.inf
    for fam, info in gmm_fits.items():
        if info["objective"] < best_obj:
            best_obj = info["objective"]
            best_family = fam

    # KL comparison with quantile-matching fits
    span = eq.upper - eq.lower
    grid = np.linspace(eq.lower - 0.3 * span, eq.upper + 0.3 * span, 300)
    emp_pdf = _empirical_pdf_from_quantiles(eq, grid)

    qm_fits = fit_all_distributions(eq)
    kl_comparison = {}

    for fam, ginfo in gmm_fits.items():
        # Build GMM distribution's PDF on grid
        p = ginfo["params"]
        if fam == "normal":
            gmm_pdf = stats.norm.pdf(grid, loc=p[0], scale=p[1])
        elif fam == "lognormal":
            gmm_pdf = stats.lognorm.pdf(grid, s=p[1], scale=np.exp(p[0]))
        elif fam == "gamma":
            gmm_pdf = stats.gamma.pdf(grid, a=p[0], scale=1.0 / p[1])
        else:
            continue

        gmm_kl = _kl_on_grid(emp_pdf, gmm_pdf, grid)

        # Find matching QM fit
        qm_match = [f for f in qm_fits if f.family == fam]
        if qm_match:
            qm_pdf_vals = np.interp(grid, qm_match[0].x_grid, qm_match[0].pdf_values)
            qm_kl = _kl_on_grid(emp_pdf, qm_pdf_vals, grid)
        else:
            qm_kl = np.inf

        kl_comparison[fam] = {"gmm_kl": float(gmm_kl), "qm_kl": float(qm_kl)}

    # Decide overall best method
    gmm_wins = 0
    qm_wins = 0
    for fam, comp in kl_comparison.items():
        if comp["gmm_kl"] < comp["qm_kl"]:
            gmm_wins += 1
        else:
            qm_wins += 1
    overall_best = "gmm" if gmm_wins > qm_wins else "quantile_matching"

    return {
        "moment_estimates": moments,
        "gmm_fits": gmm_fits,
        "best_gmm_family": best_family,
        "kl_gmm_vs_quantile": kl_comparison,
        "overall_best_method": overall_best,
    }
