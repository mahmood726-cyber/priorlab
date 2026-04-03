"""Bayesian Bootstrap for Nonparametric Prior Construction.

Constructs a nonparametric prior directly from expert quantile statements
using the Bayesian bootstrap (Rubin 1981) with kernel density smoothing.
"""

import numpy as np
from priorlab.models import ElicitedQuantiles


def _gaussian_kernel(x, center, bandwidth):
    """Unnormalized Gaussian kernel evaluated at array x."""
    z = (x - center) / bandwidth
    return np.exp(-0.5 * z ** 2) / (bandwidth * np.sqrt(2.0 * np.pi))


def bayesian_bootstrap_density(quantiles, grid=None, n_bootstrap=500,
                                bandwidth=None, seed=42):
    """Construct nonparametric density via Bayesian bootstrap.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Expert quantile elicitation.
    grid : array-like or None
        Evaluation grid. If None, auto-generated from quantile range.
    n_bootstrap : int
        Number of bootstrap samples (default 500).
    bandwidth : float or None
        Gaussian kernel bandwidth. If None, (upper - lower) / 10.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: nonparametric_density, credible_lower, credible_upper,
    grid, bandwidth.
    """
    rng = np.random.default_rng(seed)

    # Atoms at the 5 elicited quantile values
    atoms = np.array([quantiles.lower, quantiles.q1, quantiles.median,
                      quantiles.q3, quantiles.upper])
    k = len(atoms)

    if bandwidth is None:
        bandwidth = (quantiles.upper - quantiles.lower) / 10.0
    bandwidth = max(bandwidth, 1e-8)

    if grid is None:
        span = quantiles.upper - quantiles.lower
        lo = quantiles.lower - 0.3 * span
        hi = quantiles.upper + 0.3 * span
        grid = np.linspace(lo, hi, 200)
    else:
        grid = np.asarray(grid, dtype=float)

    # Bootstrap: draw Dirichlet(1,...,1) weights, smooth with Gaussian kernel
    all_densities = np.zeros((n_bootstrap, len(grid)))

    for b in range(n_bootstrap):
        # Dirichlet(1,1,...,1) weights for k atoms
        weights = rng.dirichlet(np.ones(k))

        # Kernel density: weighted sum of Gaussian kernels at atoms
        density = np.zeros_like(grid)
        for j in range(k):
            density += weights[j] * _gaussian_kernel(grid, atoms[j], bandwidth)

        all_densities[b] = density

    # Nonparametric density: mean of bootstrap densities
    mean_density = all_densities.mean(axis=0)

    # Normalize to integrate to 1
    integral = np.trapezoid(mean_density, grid)
    if integral > 0:
        mean_density /= integral
        all_densities /= integral

    # Credible bands: pointwise 2.5% and 97.5% quantiles
    credible_lower = np.percentile(all_densities, 2.5, axis=0)
    credible_upper = np.percentile(all_densities, 97.5, axis=0)

    return {
        "nonparametric_density": mean_density.tolist(),
        "credible_lower": credible_lower.tolist(),
        "credible_upper": credible_upper.tolist(),
        "grid": grid.tolist(),
        "bandwidth": bandwidth,
    }


def concentration_rate(quantiles, seed=42):
    """Measure posterior concentration as number of quantile points increases.

    Simulates bootstrap with 3, 4, 5, 7, 9 quantile points and measures
    Frechet variance of bootstrap densities at each.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Expert quantile elicitation (provides the atom pool).
    seed : int
        Random seed.

    Returns
    -------
    list of dict with keys: n_quantiles, variance.
    """
    rng = np.random.default_rng(seed)

    # Full atom pool with interpolated points
    base_atoms = [quantiles.lower, quantiles.q1, quantiles.median,
                  quantiles.q3, quantiles.upper]

    # Generate sub-selections of different sizes
    atom_configs = {
        3: [base_atoms[0], base_atoms[2], base_atoms[4]],  # lower, median, upper
        4: [base_atoms[0], base_atoms[1], base_atoms[3], base_atoms[4]],
        5: base_atoms[:],
        7: sorted(base_atoms + [
            0.5 * (base_atoms[0] + base_atoms[1]),
            0.5 * (base_atoms[3] + base_atoms[4]),
        ]),
        9: sorted(base_atoms + [
            0.5 * (base_atoms[0] + base_atoms[1]),
            0.5 * (base_atoms[1] + base_atoms[2]),
            0.5 * (base_atoms[2] + base_atoms[3]),
            0.5 * (base_atoms[3] + base_atoms[4]),
        ]),
    }

    span = quantiles.upper - quantiles.lower
    grid = np.linspace(quantiles.lower - 0.3 * span,
                       quantiles.upper + 0.3 * span, 200)
    bandwidth = span / 10.0
    bandwidth = max(bandwidth, 1e-8)
    n_boot = 200

    results = []
    for n_q, atoms in sorted(atom_configs.items()):
        atoms_arr = np.array(atoms)
        k = len(atoms_arr)
        densities = np.zeros((n_boot, len(grid)))

        for b in range(n_boot):
            weights = rng.dirichlet(np.ones(k))
            density = np.zeros_like(grid)
            for j in range(k):
                density += weights[j] * _gaussian_kernel(grid, atoms_arr[j], bandwidth)
            # Normalize
            integral = np.trapezoid(density, grid)
            if integral > 0:
                density /= integral
            densities[b] = density

        # Frechet variance: mean squared L2 distance to mean density
        mean_d = densities.mean(axis=0)
        sq_dists = []
        for b in range(n_boot):
            diff = densities[b] - mean_d
            sq_dist = np.trapezoid(diff ** 2, grid)
            sq_dists.append(sq_dist)
        fvar = float(np.mean(sq_dists))

        results.append({"n_quantiles": n_q, "variance": fvar})

    return results


def kl_vs_parametric(quantiles, parametric_pdf, grid=None, n_bootstrap=500,
                     seed=42):
    """KL divergence between nonparametric bootstrap prior and parametric fit.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Expert quantile elicitation.
    parametric_pdf : array-like
        Parametric density evaluated on *grid*.
    grid : array-like or None
        Evaluation grid (must match parametric_pdf).
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    float: KL(nonparametric || parametric).
    """
    result = bayesian_bootstrap_density(quantiles, grid=grid,
                                         n_bootstrap=n_bootstrap, seed=seed)
    p = np.asarray(result["nonparametric_density"], dtype=float)
    q = np.asarray(parametric_pdf, dtype=float)
    g = np.asarray(result["grid"], dtype=float)

    epsilon = 1e-300
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)

    integrand = p * np.log(p / q)
    kl = float(np.trapezoid(integrand, g))
    return kl


def full_bootstrap_analysis(quantiles, parametric_pdf=None, grid=None,
                            n_bootstrap=500, seed=42):
    """Full Bayesian bootstrap prior analysis.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Expert quantile elicitation.
    parametric_pdf : array-like or None
        Best parametric fit density on grid (for KL comparison).
    grid : array-like or None
        Evaluation grid.
    n_bootstrap : int
        Number of bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: nonparametric_density, credible_lower, credible_upper,
    grid, concentration_rate, kl_vs_parametric, bandwidth.
    """
    boot = bayesian_bootstrap_density(quantiles, grid=grid,
                                       n_bootstrap=n_bootstrap, seed=seed)

    conc = concentration_rate(quantiles, seed=seed)

    kl = None
    if parametric_pdf is not None:
        kl = kl_vs_parametric(quantiles, parametric_pdf,
                              grid=np.asarray(boot["grid"]),
                              n_bootstrap=n_bootstrap, seed=seed)

    return {
        "nonparametric_density": boot["nonparametric_density"],
        "credible_lower": boot["credible_lower"],
        "credible_upper": boot["credible_upper"],
        "grid": boot["grid"],
        "concentration_rate": conc,
        "kl_vs_parametric": kl,
        "bandwidth": boot["bandwidth"],
    }
