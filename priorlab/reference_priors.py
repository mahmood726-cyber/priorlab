"""Non-informative and reference priors for comparison with elicited priors.

Provides Jeffreys priors, maximum entropy priors, and information-theoretic
comparison metrics (KL divergence, information gain) to benchmark elicited
priors against principled non-informative alternatives.
"""

import numpy as np
from scipy.stats import norm as _norm

from priorlab.divergence import kl_divergence


def jeffreys_normal_mean(grid, sigma2):
    """Jeffreys prior for Normal mean (known variance): p(mu) proportional to 1.

    Flat (improper) prior, normalized over the finite grid.

    Parameters
    ----------
    grid : array-like
        Evaluation points.
    sigma2 : float
        Known variance (not used in density, but kept for API clarity).

    Returns
    -------
    numpy.ndarray
        Normalized density on the grid.
    """
    grid = np.asarray(grid, dtype=float)
    density = np.ones_like(grid)
    integral = np.trapezoid(density, grid)
    if integral > 0:
        density /= integral
    return density


def jeffreys_normal_variance(grid):
    """Jeffreys prior for Normal variance (known mean): p(sigma2) proportional to 1/sigma2.

    Normalized over the finite positive grid.

    Parameters
    ----------
    grid : array-like
        Evaluation points (should be positive).

    Returns
    -------
    numpy.ndarray
        Normalized density on the grid.
    """
    grid = np.asarray(grid, dtype=float)
    density = np.where(grid > 0, 1.0 / grid, 0.0)
    integral = np.trapezoid(density, grid)
    if integral > 0:
        density /= integral
    return density


def jeffreys_normal_joint_marginals(mu_grid, sigma2_grid):
    """Jeffreys prior for Normal (mu, sigma2): p(mu, sigma2) proportional to 1/sigma2.

    Returns marginal densities for mu and sigma2 independently.

    Parameters
    ----------
    mu_grid : array-like
        Evaluation points for mu.
    sigma2_grid : array-like
        Evaluation points for sigma2 (positive).

    Returns
    -------
    dict with keys: marginal_mu (flat, normalized), marginal_sigma2 (1/sigma2, normalized).
    """
    mu_grid = np.asarray(mu_grid, dtype=float)
    sigma2_grid = np.asarray(sigma2_grid, dtype=float)

    # Marginal for mu: integrate out sigma2 from 1/sigma2 -> flat in mu
    marginal_mu = np.ones_like(mu_grid)
    integral_mu = np.trapezoid(marginal_mu, mu_grid)
    if integral_mu > 0:
        marginal_mu /= integral_mu

    # Marginal for sigma2: integrate out mu from 1/sigma2 -> 1/sigma2
    marginal_sigma2 = np.where(sigma2_grid > 0, 1.0 / sigma2_grid, 0.0)
    integral_s2 = np.trapezoid(marginal_sigma2, sigma2_grid)
    if integral_s2 > 0:
        marginal_sigma2 /= integral_s2

    return {
        "marginal_mu": marginal_mu,
        "marginal_sigma2": marginal_sigma2,
    }


def maxent_moments(grid, mu_constraint, second_moment_constraint):
    """Maximum entropy distribution under mean and second moment constraints.

    Under constraints E[x] = mu, E[x^2] = m2, the MaxEnt distribution is
    p(x) proportional to exp(lam0 + lam1*x + lam2*x^2), which is Gaussian with
    mu = mu_constraint, sigma^2 = m2 - mu^2.

    Parameters
    ----------
    grid : array-like
        Evaluation points.
    mu_constraint : float
        Constraint on E[x].
    second_moment_constraint : float
        Constraint on E[x^2]. Must be > mu_constraint^2.

    Returns
    -------
    numpy.ndarray
        Normalized MaxEnt density on the grid.
    """
    grid = np.asarray(grid, dtype=float)
    var = second_moment_constraint - mu_constraint ** 2
    if var <= 0:
        raise ValueError(
            "second_moment_constraint must exceed mu_constraint^2 "
            "to yield positive variance"
        )
    sigma = np.sqrt(var)
    density = _norm(loc=mu_constraint, scale=sigma).pdf(grid)
    return density


def maxent_support(grid, a, b, mean_constraint=None, var_constraint=None):
    """Maximum entropy distribution with bounded support [a, b].

    - No constraints: Uniform(a, b)
    - Mean constraint only: truncated exponential
    - Mean + variance: truncated Normal

    Parameters
    ----------
    grid : array-like
        Evaluation points.
    a, b : float
        Support bounds (a < b).
    mean_constraint : float or None
        Optional constraint on E[x].
    var_constraint : float or None
        Optional constraint on Var[x]. Requires mean_constraint.

    Returns
    -------
    numpy.ndarray
        Normalized density on the grid.
    """
    grid = np.asarray(grid, dtype=float)
    if a >= b:
        raise ValueError("a must be less than b")

    mask = (grid >= a) & (grid <= b)

    if mean_constraint is None and var_constraint is None:
        # No constraints -> Uniform(a, b)
        density = np.where(mask, 1.0 / (b - a), 0.0)
        return density

    if mean_constraint is not None and var_constraint is None:
        # Mean constraint only -> truncated exponential
        # p(x) proportional to exp(lam * x) on [a, b]
        # Solve for lam such that E[x] = mean_constraint
        # Use Newton's method on the mean equation
        midpoint = (a + b) / 2.0
        if abs(mean_constraint - midpoint) < 1e-10:
            # lam ~ 0, uniform
            density = np.where(mask, 1.0 / (b - a), 0.0)
            return density

        lam = _solve_exponential_lambda(a, b, mean_constraint)
        density = np.where(mask, np.exp(lam * grid), 0.0)
        integral = np.trapezoid(density, grid)
        if integral > 0:
            density /= integral
        return density

    if mean_constraint is not None and var_constraint is not None:
        # Mean + variance -> truncated Normal
        if var_constraint <= 0:
            raise ValueError("var_constraint must be positive")
        sigma = np.sqrt(var_constraint)
        raw = _norm(loc=mean_constraint, scale=sigma).pdf(grid)
        density = np.where(mask, raw, 0.0)
        integral = np.trapezoid(density, grid)
        if integral > 0:
            density /= integral
        return density

    raise ValueError("var_constraint requires mean_constraint to be set")


def _solve_exponential_lambda(a, b, target_mean, max_iter=100, tol=1e-10):
    """Solve for lambda in truncated exponential so that E[x] = target_mean.

    Uses Newton's method on f(lam) = E_lam[x] - target_mean.
    """
    lam = 0.0
    for _ in range(max_iter):
        if abs(lam) < 1e-12:
            # Taylor expansion near lam=0
            current_mean = (a + b) / 2.0
            # Derivative of mean w.r.t. lam at lam=0 is (b-a)^2/12
            deriv = (b - a) ** 2 / 12.0
        else:
            ea = np.exp(lam * a)
            eb = np.exp(lam * b)
            # Mean = d/dlam log(Z), Z = (e^(lam*b) - e^(lam*a))/lam
            # Mean = (b*e^(lam*b) - a*e^(lam*a))/(e^(lam*b)-e^(lam*a)) - 1/lam
            denom = eb - ea
            if abs(denom) < 1e-300:
                break
            current_mean = (b * eb - a * ea) / denom - 1.0 / lam
            # Derivative: Var[x] under this distribution
            e2 = (b**2 * eb - a**2 * ea) / denom
            deriv = e2 - current_mean**2

        err = current_mean - target_mean
        if abs(err) < tol:
            break
        if abs(deriv) < 1e-300:
            break
        lam -= err / deriv

    return lam


def information_gain(prior_pdf, posterior_pdf, grid):
    """Information gain: KL(posterior || prior).

    Measures how much the prior "learns" from data.

    Parameters
    ----------
    prior_pdf, posterior_pdf : array-like
        Density arrays on the same grid.
    grid : array-like
        Evaluation grid.

    Returns
    -------
    float
        KL(posterior || prior) in nats.
    """
    return kl_divergence(posterior_pdf, prior_pdf, grid)


def compare_reference_priors(elicited_pdf, grid, sigma2=1.0,
                             mu_constraint=None, second_moment=None,
                             posterior_pdf=None):
    """Compare an elicited prior against reference priors.

    Parameters
    ----------
    elicited_pdf : array-like
        Elicited prior density on the grid.
    grid : array-like
        Evaluation grid.
    sigma2 : float
        Known variance for Jeffreys mean prior.
    mu_constraint : float or None
        Mean constraint for MaxEnt. Defaults to grid midpoint.
    second_moment : float or None
        Second moment for MaxEnt. Defaults to mu^2 + grid_range^2/12.
    posterior_pdf : array-like or None
        Posterior density for information gain computation.

    Returns
    -------
    dict with keys: jeffreys_mean, jeffreys_variance, maxent_pdf, uniform_pdf,
    kl_elicited_vs_jeffreys, kl_elicited_vs_maxent, kl_elicited_vs_uniform,
    info_gain_elicited, info_gain_jeffreys, info_gain_maxent.
    """
    grid = np.asarray(grid, dtype=float)
    elicited_pdf = np.asarray(elicited_pdf, dtype=float)

    # Jeffreys priors
    jeff_mean = jeffreys_normal_mean(grid, sigma2)
    jeff_var = jeffreys_normal_variance(np.maximum(grid, 1e-12))

    # MaxEnt prior
    a, b = float(grid[0]), float(grid[-1])
    if mu_constraint is None:
        mu_constraint = (a + b) / 2.0
    if second_moment is None:
        second_moment = mu_constraint ** 2 + (b - a) ** 2 / 12.0

    maxent_pdf = maxent_moments(grid, mu_constraint, second_moment)

    # Uniform prior on grid support
    uniform_pdf = np.ones_like(grid) / (b - a) if b > a else np.ones_like(grid)

    # KL divergences: KL(elicited || reference)
    kl_vs_jeffreys = kl_divergence(elicited_pdf, jeff_mean, grid)
    kl_vs_maxent = kl_divergence(elicited_pdf, maxent_pdf, grid)
    kl_vs_uniform = kl_divergence(elicited_pdf, uniform_pdf, grid)

    # Information gain: KL(posterior || prior) for each
    ig_elicited = 0.0
    ig_jeffreys = 0.0
    ig_maxent = 0.0
    if posterior_pdf is not None:
        posterior_pdf = np.asarray(posterior_pdf, dtype=float)
        ig_elicited = information_gain(elicited_pdf, posterior_pdf, grid)
        ig_jeffreys = information_gain(jeff_mean, posterior_pdf, grid)
        ig_maxent = information_gain(maxent_pdf, posterior_pdf, grid)

    return {
        "jeffreys_mean": jeff_mean.tolist(),
        "jeffreys_variance": jeff_var.tolist(),
        "maxent_pdf": maxent_pdf.tolist(),
        "uniform_pdf": uniform_pdf.tolist(),
        "kl_elicited_vs_jeffreys": float(kl_vs_jeffreys),
        "kl_elicited_vs_maxent": float(kl_vs_maxent),
        "kl_elicited_vs_uniform": float(kl_vs_uniform),
        "info_gain_elicited": float(ig_elicited),
        "info_gain_jeffreys": float(ig_jeffreys),
        "info_gain_maxent": float(ig_maxent),
    }
