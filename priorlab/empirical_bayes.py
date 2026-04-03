"""Empirical Bayes estimation for PriorLab.

Estimate hyperparameters from the data itself using multiple experts'
quantile-derived estimates.  Implements parametric EB (Normal-Normal),
James-Stein shrinkage, marginal likelihood profiling, and SURE.
"""

import numpy as np
from scipy import stats

from priorlab.fitting import fit_normal
from priorlab.models import ExpertPrior


def _extract_expert_summaries(experts):
    """Extract (mu_i, sigma_i) pairs from each expert's fitted Normal.

    If an expert has no fitted distribution we fit one from their quantiles.
    Returns parallel lists: ids, mus, sigmas.
    """
    ids, mus, sigmas = [], [], []
    for exp in experts:
        fit = exp.best_fit
        if fit is None and exp.quantiles is not None:
            fit = fit_normal(exp.quantiles)
        if fit is None:
            continue
        mu = fit.params.get("mu") or fit.params.get("loc", 0.0)
        sigma = fit.params.get("sigma") or fit.params.get("scale", 1.0)
        ids.append(exp.expert_id)
        mus.append(mu)
        sigmas.append(sigma)
    return ids, np.asarray(mus, dtype=float), np.asarray(sigmas, dtype=float)


def empirical_bayes_analysis(experts, grid_size=50):
    """Run full parametric empirical Bayes analysis.

    Parameters
    ----------
    experts : list[ExpertPrior]
        At least 3 experts, each with quantiles and/or a fitted distribution.
    grid_size : int
        Number of grid points for the marginal likelihood profile over A.

    Returns
    -------
    dict with keys:
        grand_mean, between_variance_A, shrinkage_factor_B,
        shrunk_estimates (dict expert_id -> float),
        eb_posterior_means (dict), eb_posterior_ses (dict),
        marginal_ll_profile (list of {A, ll}),
        sure_mse_estimate (float)
    """
    ids, mus, sigmas = _extract_expert_summaries(experts)
    p = len(mus)
    if p < 3:
        raise ValueError("James-Stein shrinkage requires p >= 3 experts")

    sigma_sq = sigmas ** 2

    # ---- Method-of-moments estimates ----
    mu_0 = float(np.mean(mus))
    A_mom = max(0.0, float(np.var(mus, ddof=0) - np.mean(sigma_sq)))

    # ---- James-Stein shrinkage ----
    ss = float(np.sum((mus - mu_0) ** 2))
    if ss < 1e-30:
        B = 1.0  # all experts identical -> full shrinkage
    else:
        B = float((p - 2) * np.mean(sigma_sq) / ss)
    B = float(np.clip(B, 0.0, 1.0))

    shrunk = mu_0 + (1.0 - B) * (mus - mu_0)
    shrunk_dict = {eid: float(s) for eid, s in zip(ids, shrunk)}

    # ---- Empirical Bayes posterior (Normal-Normal) ----
    eb_means = {}
    eb_ses = {}
    for i, eid in enumerate(ids):
        w = sigma_sq[i] / (sigma_sq[i] + A_mom) if (sigma_sq[i] + A_mom) > 0 else 1.0
        eb_mu = w * mu_0 + (1.0 - w) * mus[i]
        eb_var = 1.0 / (1.0 / (sigma_sq[i] + 1e-30) + 1.0 / (A_mom + 1e-30)) if A_mom > 0 else sigma_sq[i]
        eb_means[eid] = float(eb_mu)
        eb_ses[eid] = float(np.sqrt(max(eb_var, 0.0)))

    # ---- Marginal likelihood profile over A ----
    A_max = max(float(np.var(mus, ddof=0)) * 5, 1.0)
    A_grid = np.linspace(0, A_max, grid_size)
    profile = []
    for A_val in A_grid:
        ll = 0.0
        for i in range(p):
            total_var = sigma_sq[i] + A_val
            ll += stats.norm.logpdf(mus[i], loc=mu_0, scale=np.sqrt(total_var))
        profile.append({"A": float(A_val), "ll": float(ll)})

    # MLE of A
    best_idx = int(np.argmax([pt["ll"] for pt in profile]))
    A_mle = profile[best_idx]["A"]

    # ---- SURE (Stein's Unbiased Risk Estimate) ----
    # For the James-Stein estimator, SURE = sum(sigma_i^2)
    #   - 2 * sum(sigma_i^2 * d(shrink_i)/d(mu_i))
    # d(shrink_i)/d(mu_i) = (1-B) + (mu_i-mu_0)^2 * 2*(p-2)*mean(sigma^2)/ss^2
    # but the second term adjusts for the dependence of B on mu_i.
    # Simplified SURE for JS:
    if ss > 1e-30:
        mean_s2 = float(np.mean(sigma_sq))
        # derivative of shrunk_i w.r.t. mu_i
        # shrunk_i = mu_0 + (1 - B) * (mu_i - mu_0)
        # B = (p-2)*mean_s2 / ss,  ss = sum((mu_j - mu_0)^2)
        # d(shrunk_i)/d(mu_i) = (1-B) + (mu_i-mu_0)*d(1-B)/d(mu_i)
        # d(B)/d(mu_i) = -(p-2)*mean_s2 * 2*(mu_i-mu_0) / ss^2
        d_shrink = np.zeros(p)
        for i in range(p):
            dB_dmu = -(p - 2) * mean_s2 * 2 * (mus[i] - mu_0) / (ss ** 2)
            d_shrink[i] = (1 - B) - (mus[i] - mu_0) * dB_dmu
        sure = float(np.mean(sigma_sq) + (2.0 / p) * np.sum(sigma_sq * d_shrink))
    else:
        sure = float(np.mean(sigma_sq))

    return {
        "grand_mean": mu_0,
        "between_variance_A": A_mom,
        "shrinkage_factor_B": B,
        "shrunk_estimates": shrunk_dict,
        "eb_posterior_means": eb_means,
        "eb_posterior_ses": eb_ses,
        "marginal_ll_profile": profile,
        "sure_mse_estimate": sure,
    }
