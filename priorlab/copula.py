"""Gaussian copula for joint prior elicitation.

Model dependence between two parameters (e.g., treatment effect
and heterogeneity tau^2) via a Gaussian copula that preserves
arbitrary marginals while adding correlation.
"""

import numpy as np
from scipy import stats
from scipy.integrate import cumulative_trapezoid


def _marginal_cdf(grid, pdf):
    """Compute CDF from a discrete PDF via cumulative trapezoid."""
    pdf = np.asarray(pdf, dtype=float)
    grid = np.asarray(grid, dtype=float)
    cdf = np.concatenate(([0.0], cumulative_trapezoid(pdf, grid)))
    # Normalize to [0, 1]
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    # Clamp to valid range
    cdf = np.clip(cdf, 0.0, 1.0)
    return cdf


def _safe_ppf(u, clip_lo=1e-10, clip_hi=1 - 1e-10):
    """Phi^{-1}(u) with safe clamping to avoid +/-inf."""
    u = np.clip(u, clip_lo, clip_hi)
    return stats.norm.ppf(u)


def kendall_tau(rho):
    """Kendall's tau from Gaussian copula correlation rho.

    tau = (2/pi) * arcsin(rho)
    """
    return float(2.0 / np.pi * np.arcsin(rho))


def joint_density(grid1, pdf1, grid2, pdf2, rho=0.0):
    """Compute 2D joint density via Gaussian copula.

    Parameters
    ----------
    grid1 : array-like
        Grid for parameter 1.
    pdf1 : array-like
        Marginal PDF for parameter 1.
    grid2 : array-like
        Grid for parameter 2.
    pdf2 : array-like
        Marginal PDF for parameter 2.
    rho : float
        Copula correlation in (-1, 1).

    Returns
    -------
    ndarray of shape (len(grid1), len(grid2))
        Joint density values.
    """
    grid1 = np.asarray(grid1, dtype=float)
    grid2 = np.asarray(grid2, dtype=float)
    pdf1 = np.asarray(pdf1, dtype=float)
    pdf2 = np.asarray(pdf2, dtype=float)

    cdf1 = _marginal_cdf(grid1, pdf1)
    cdf2 = _marginal_cdf(grid2, pdf2)

    z1 = _safe_ppf(cdf1)
    z2 = _safe_ppf(cdf2)

    n1 = len(grid1)
    n2 = len(grid2)
    joint = np.zeros((n1, n2))

    if abs(rho) < 1e-12:
        # Independence: joint = f1(x1) * f2(x2)
        for i in range(n1):
            for j in range(n2):
                joint[i, j] = pdf1[i] * pdf2[j]
    else:
        rho_sq = rho ** 2
        det_factor = np.sqrt(1.0 - rho_sq)
        for i in range(n1):
            for j in range(n2):
                # Copula density c(u1, u2)
                # = phi_2(z1, z2; rho) / (phi(z1) * phi(z2))
                # phi_2 = bivariate normal pdf
                zi = z1[i]
                zj = z2[j]
                exponent = (zi ** 2 + zj ** 2 - 2 * rho * zi * zj) / (2 * (1 - rho_sq))
                phi2 = np.exp(-exponent) / (2 * np.pi * det_factor)
                phi_zi = stats.norm.pdf(zi)
                phi_zj = stats.norm.pdf(zj)
                denom = phi_zi * phi_zj
                if denom > 1e-300:
                    copula_density = phi2 / denom
                else:
                    copula_density = 1.0
                joint[i, j] = copula_density * pdf1[i] * pdf2[j]

    return joint


def conditional_pdf(grid1, pdf1, grid2, pdf2, rho, x1_value=None):
    """Conditional PDF f(x2 | x1) via Gaussian copula.

    If x1_value is None, uses the median of marginal 1.

    Conditional CDF: F(x2|x1) = Phi((z2 - rho*z1) / sqrt(1 - rho^2))
    Conditional PDF: differentiate numerically.

    Parameters
    ----------
    grid1, pdf1 : array-like
        Marginal 1.
    grid2, pdf2 : array-like
        Marginal 2.
    rho : float
        Copula correlation.
    x1_value : float, optional
        Conditioning value for x1.

    Returns
    -------
    ndarray of conditional PDF values on grid2.
    """
    grid1 = np.asarray(grid1, dtype=float)
    grid2 = np.asarray(grid2, dtype=float)
    pdf1 = np.asarray(pdf1, dtype=float)
    pdf2 = np.asarray(pdf2, dtype=float)

    cdf1 = _marginal_cdf(grid1, pdf1)
    cdf2 = _marginal_cdf(grid2, pdf2)

    # Find z1 at conditioning point
    if x1_value is None:
        # Use median: find grid point closest to CDF = 0.5
        idx1 = np.argmin(np.abs(cdf1 - 0.5))
    else:
        idx1 = np.argmin(np.abs(grid1 - x1_value))

    u1 = cdf1[idx1]
    z1_val = _safe_ppf(np.array([u1]))[0]

    z2 = _safe_ppf(cdf2)

    if abs(rho) < 1e-12:
        # Independence: conditional = marginal
        return pdf2.copy()

    sqrt_1_rho2 = np.sqrt(1.0 - rho ** 2)

    # Conditional CDF: Phi((z2 - rho*z1) / sqrt(1-rho^2))
    cond_cdf = stats.norm.cdf((z2 - rho * z1_val) / sqrt_1_rho2)

    # Numerical differentiation for PDF
    cond_pdf = np.gradient(cond_cdf, grid2)
    cond_pdf = np.maximum(cond_pdf, 0.0)

    # Normalize
    integral = np.trapezoid(cond_pdf, grid2)
    if integral > 1e-12:
        cond_pdf /= integral

    return cond_pdf


def gaussian_copula(grid1, pdf1, grid2, pdf2, rho=0.0):
    """Full Gaussian copula analysis for two marginal priors.

    Parameters
    ----------
    grid1 : array-like
        Grid for parameter 1.
    pdf1 : array-like
        Marginal PDF for parameter 1.
    grid2 : array-like
        Grid for parameter 2.
    pdf2 : array-like
        Marginal PDF for parameter 2.
    rho : float
        Copula correlation in (-1, 1).  Default 0 (independence).

    Returns
    -------
    dict with keys: joint_density (2D list), conditional_pdf_given_x1 (list),
    kendall_tau, marginal1_cdf (list), marginal2_cdf (list), rho.
    """
    grid1 = np.asarray(grid1, dtype=float)
    grid2 = np.asarray(grid2, dtype=float)
    pdf1 = np.asarray(pdf1, dtype=float)
    pdf2 = np.asarray(pdf2, dtype=float)

    jd = joint_density(grid1, pdf1, grid2, pdf2, rho)
    cond = conditional_pdf(grid1, pdf1, grid2, pdf2, rho)
    tau = kendall_tau(rho)
    cdf1 = _marginal_cdf(grid1, pdf1)
    cdf2 = _marginal_cdf(grid2, pdf2)

    return {
        "joint_density": jd.tolist(),
        "conditional_pdf_given_x1": cond.tolist(),
        "kendall_tau": tau,
        "marginal1_cdf": cdf1.tolist(),
        "marginal2_cdf": cdf2.tolist(),
        "rho": float(rho),
    }
