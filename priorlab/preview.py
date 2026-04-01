import numpy as np
from scipy.stats import norm


def conjugate_normal_update(mu_prior, sigma_prior, theta_data, se_data):
    """Conjugate Normal-Normal posterior update."""
    prec_prior = 1.0 / (sigma_prior ** 2)
    prec_data = 1.0 / (se_data ** 2)
    prec_post = prec_prior + prec_data
    sigma_post = 1.0 / np.sqrt(prec_post)
    mu_post = (prec_prior * mu_prior + prec_data * theta_data) / prec_post
    shrinkage = 1.0 - (prec_prior / prec_post)
    return {
        "mu_post": float(mu_post),
        "sigma_post": float(sigma_post),
        "shrinkage": float(shrinkage),
        "ci_lo": float(mu_post - 1.96 * sigma_post),
        "ci_hi": float(mu_post + 1.96 * sigma_post),
    }


def grid_posterior(grid, prior_pdf, theta_data, se_data):
    """Numerical grid posterior for arbitrary priors."""
    likelihood = norm(loc=theta_data, scale=se_data).pdf(grid)
    unnorm = np.array(prior_pdf) * likelihood
    integral = np.trapezoid(unnorm, grid)
    if integral > 0:
        return unnorm / integral
    return unnorm
