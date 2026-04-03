"""Conjugate prior analysis for common likelihood-prior pairs.

Provides posterior computation, predictive distributions, shrinkage factors,
KL divergence, and effective sample sizes for:
  - Normal-Normal (known variance)
  - Normal-InverseGamma (unknown mean and variance)
  - Beta-Binomial
  - Gamma-Poisson
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Normal-Normal (known variance)
# ---------------------------------------------------------------------------

def normal_normal(mu0, sigma0, y=0.0, sigma_y=1.0):
    """Conjugate Normal-Normal update (known likelihood variance).

    Prior:      theta ~ N(mu0, sigma0^2)
    Likelihood: y | theta ~ N(theta, sigma_y^2)

    Returns dict with posterior, predictive, shrinkage, kl, ess.
    """
    tau0 = 1.0 / sigma0 ** 2   # prior precision
    tau_y = 1.0 / sigma_y ** 2  # data precision

    tau_n = tau0 + tau_y
    sigma_n = np.sqrt(1.0 / tau_n)
    mu_n = (tau0 * mu0 + tau_y * y) / tau_n

    # Shrinkage: how much of the posterior mean comes from the prior
    shrinkage = tau0 / tau_n  # fraction from prior

    # Predictive: y_new | y ~ N(mu_n, sigma_n^2 + sigma_y^2)
    pred_var = sigma_n ** 2 + sigma_y ** 2
    pred_sd = np.sqrt(pred_var)

    # KL(posterior || prior)
    kl = 0.5 * ((sigma_n / sigma0) ** 2 + (mu_n - mu0) ** 2 / sigma0 ** 2 - 1
                 + 2 * np.log(sigma0 / sigma_n))

    # Effective prior sample size
    ess = sigma_y ** 2 / sigma0 ** 2

    # PDF grids for plotting
    span = max(4 * sigma0, 4 * sigma_y)
    grid = np.linspace(min(mu0, y, mu_n) - span, max(mu0, y, mu_n) + span, 200)
    prior_pdf = stats.norm.pdf(grid, mu0, sigma0).tolist()
    post_pdf = stats.norm.pdf(grid, mu_n, sigma_n).tolist()
    pred_pdf = stats.norm.pdf(grid, mu_n, pred_sd).tolist()

    return {
        "posterior_mean": float(mu_n),
        "posterior_sd": float(sigma_n),
        "predictive_mean": float(mu_n),
        "predictive_sd": float(pred_sd),
        "shrinkage": float(shrinkage),
        "kl_post_prior": float(kl),
        "effective_sample_size": float(ess),
        "posterior_precision": float(tau_n),
        "prior_precision": float(tau0),
        "grid": grid.tolist(),
        "prior_pdf": prior_pdf,
        "posterior_pdf": post_pdf,
        "predictive_pdf": pred_pdf,
    }


# ---------------------------------------------------------------------------
# 2. Normal-InverseGamma (unknown mean and variance)
# ---------------------------------------------------------------------------

def normal_inverse_gamma(mu0, kappa0, alpha0, beta0, y=0.0, n=1):
    """Conjugate Normal-InverseGamma update.

    Prior:  sigma^2 ~ IG(alpha0, beta0)
            theta | sigma^2 ~ N(mu0, sigma^2/kappa0)
    Data:   y_bar (sample mean), n observations

    Posterior: NIG(mu_n, kappa_n, alpha_n, beta_n)
    Marginal posterior for mu: Student-t(2*alpha_n) shifted and scaled.
    """
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * y) / kappa_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * n * kappa0 / kappa_n * (y - mu0) ** 2

    # Marginal posterior for mu: t_{2*alpha_n}(mu_n, beta_n/(alpha_n*kappa_n))
    df_t = 2 * alpha_n
    scale_t = np.sqrt(beta_n / (alpha_n * kappa_n))

    # Posterior mean of sigma^2 (if alpha_n > 1)
    sigma2_post_mean = float(beta_n / (alpha_n - 1)) if alpha_n > 1 else float("inf")

    return {
        "mu_n": float(mu_n),
        "kappa_n": float(kappa_n),
        "alpha_n": float(alpha_n),
        "beta_n": float(beta_n),
        "marginal_mu_df": float(df_t),
        "marginal_mu_scale": float(scale_t),
        "sigma2_posterior_mean": sigma2_post_mean,
    }


# ---------------------------------------------------------------------------
# 3. Beta-Binomial
# ---------------------------------------------------------------------------

def beta_binomial(a, b, x=0, n=1):
    """Conjugate Beta-Binomial update.

    Prior:      p ~ Beta(a, b)
    Likelihood: x | p ~ Binomial(n, p)
    Posterior:  p | x ~ Beta(a + x, b + n - x)
    """
    a_post = a + x
    b_post = b + n - x

    prior_mean = a / (a + b)
    post_mean = a_post / (a_post + b_post)
    prior_var = a * b / ((a + b) ** 2 * (a + b + 1))
    post_var = a_post * b_post / ((a_post + b_post) ** 2 * (a_post + b_post + 1))

    # Shrinkage toward prior
    ess_prior = a + b
    shrinkage = ess_prior / (ess_prior + n)

    # KL(posterior || prior) — numerical on grid
    grid = np.linspace(1e-6, 1 - 1e-6, 500)
    prior_pdf = stats.beta.pdf(grid, a, b)
    post_pdf = stats.beta.pdf(grid, a_post, b_post)
    eps = 1e-300
    kl = float(np.trapezoid(post_pdf * np.log((post_pdf + eps) / (prior_pdf + eps)), grid))

    # Posterior predictive: Beta-Binomial(1, a_post, b_post)
    # P(x_new=1) = a_post / (a_post + b_post)
    pred_prob = a_post / (a_post + b_post)

    return {
        "a_posterior": float(a_post),
        "b_posterior": float(b_post),
        "prior_mean": float(prior_mean),
        "posterior_mean": float(post_mean),
        "prior_variance": float(prior_var),
        "posterior_variance": float(post_var),
        "prior_effective_n": float(ess_prior),
        "shrinkage": float(shrinkage),
        "kl_post_prior": float(kl),
        "predictive_prob": float(pred_prob),
        "posterior_precision": float(a_post + b_post),
        "prior_precision": float(a + b),
    }


# ---------------------------------------------------------------------------
# 4. Gamma-Poisson
# ---------------------------------------------------------------------------

def gamma_poisson(a, b, sum_x=0, n=1):
    """Conjugate Gamma-Poisson update.

    Prior:      lambda ~ Gamma(a, rate=b)
    Likelihood: sum(x_i) | lambda ~ Poisson(n*lambda)
    Posterior:  lambda | data ~ Gamma(a + sum_x, b + n)
    """
    a_post = a + sum_x
    b_post = b + n

    prior_mean = a / b
    post_mean = a_post / b_post
    prior_var = a / b ** 2
    post_var = a_post / b_post ** 2

    shrinkage = b / b_post
    ess_prior = b  # prior acts like b pseudo-observations

    # KL(posterior || prior): for Gamma distributions
    from scipy.special import gammaln, digamma
    kl = ((a_post - a) * digamma(a_post) - gammaln(a_post) + gammaln(a)
          + a * (np.log(b_post) - np.log(b)) + a_post * (b - b_post) / b_post)

    # Posterior predictive: Negative Binomial
    # P(x_new | data) = NegBin(r=a_post, p=b_post/(b_post+1))
    pred_mean = a_post / b_post  # same as posterior mean of lambda

    return {
        "a_posterior": float(a_post),
        "b_posterior": float(b_post),
        "prior_mean": float(prior_mean),
        "posterior_mean": float(post_mean),
        "prior_variance": float(prior_var),
        "posterior_variance": float(post_var),
        "shrinkage": float(shrinkage),
        "prior_effective_n": float(ess_prior),
        "kl_post_prior": float(kl),
        "predictive_mean": float(pred_mean),
        "posterior_precision": float(b_post),
        "prior_precision": float(b),
    }


# ---------------------------------------------------------------------------
# 5. Full conjugate analysis
# ---------------------------------------------------------------------------

def conjugate_analysis(eq=None, y=0.0, n=1, sigma_y=1.0):
    """Run all four conjugate analyses.

    If ``eq`` (ElicitedQuantiles) is provided, derive Normal prior parameters
    from the quantiles.  Otherwise use default weakly informative priors.

    Parameters
    ----------
    eq : ElicitedQuantiles, optional
    y : float
        Observed data point (or sample mean for NIG).
    n : int
        Sample size.
    sigma_y : float
        Likelihood standard deviation (for Normal-Normal).

    Returns
    -------
    dict with keys: normal_normal, normal_ig, beta_binomial, gamma_poisson,
                    all_posteriors_data.
    """
    # Derive priors from quantiles or use defaults
    if eq is not None:
        mu0 = eq.median
        sigma0 = (eq.q3 - eq.q1) / (2 * 0.6745)
        sigma0 = max(sigma0, 1e-6)
    else:
        mu0 = 0.0
        sigma0 = 1.0

    nn = normal_normal(mu0, sigma0, y=y, sigma_y=sigma_y)

    # Normal-IG with kappa0=1, alpha0=2, beta0=sigma0^2
    nig = normal_inverse_gamma(mu0, kappa0=1.0, alpha0=2.0,
                               beta0=sigma0 ** 2, y=y, n=n)

    # Beta-Binomial with weakly informative priors
    bb = beta_binomial(a=2.0, b=2.0, x=max(0, int(round(y))), n=max(n, 1))

    # Gamma-Poisson with weakly informative priors
    gp = gamma_poisson(a=2.0, b=1.0, sum_x=max(0, int(round(abs(y)))), n=max(n, 1))

    return {
        "normal_normal": nn,
        "normal_ig": nig,
        "beta_binomial": bb,
        "gamma_poisson": gp,
    }
