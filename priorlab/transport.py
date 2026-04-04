"""Optimal Transport for Prior Comparison.

Wasserstein-2 distance, barycenters, displacement interpolation,
and transport maps between expert prior distributions.
"""

import numpy as np
from scipy import stats
from priorlab.fitting import fit_all_distributions, _make_grid
from priorlab.models import ElicitedQuantiles, ExpertPrior


def _quantile_function(cdf_vals, grid, t_grid):
    """Compute quantile function (inverse CDF) at points t_grid.

    Parameters
    ----------
    cdf_vals : array
        CDF values on the grid.
    grid : array
        x-values corresponding to cdf_vals.
    t_grid : array
        Probability levels at which to evaluate the quantile function.

    Returns
    -------
    array
        Quantile function values at t_grid.
    """
    # Ensure CDF is monotonically non-decreasing
    cdf_vals = np.maximum.accumulate(cdf_vals)
    # Clamp to [0, 1]
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    return np.interp(t_grid, cdf_vals, grid)


def _pdf_from_quantile_fn(q_vals, t_grid, x_grid):
    """Convert a quantile function to a PDF on x_grid via histogram-like density."""
    # q_vals are quantiles at t_grid (uniform probabilities)
    # Build a density estimate by differentiating the inverse
    # dQ/dt -> 1/f(Q(t)), so f(x) = dt/dQ
    dq = np.diff(q_vals)
    dt = np.diff(t_grid)
    # Avoid division by zero
    dq_safe = np.where(np.abs(dq) < 1e-15, 1e-15, dq)
    density_at_mid = dt / dq_safe
    q_mid = 0.5 * (q_vals[:-1] + q_vals[1:])

    # Interpolate onto x_grid
    pdf = np.interp(x_grid, q_mid, density_at_mid, left=0.0, right=0.0)
    # Normalize
    area = np.trapezoid(pdf, x_grid)
    if area > 1e-15:
        pdf = pdf / area
    return pdf


def _get_cdf_pdf_grid(expert: ExpertPrior, grid):
    """Get CDF and PDF on grid from an expert's best fit."""
    fitted = expert.best_fit
    if fitted is None:
        return None, None

    family = fitted.family
    params = fitted.params

    if family == "normal":
        dist = stats.norm(loc=params["mu"], scale=params["sigma"])
    elif family == "lognormal":
        dist = stats.lognorm(s=params["sigma_log"], scale=np.exp(params["mu_log"]))
    elif family == "gamma":
        dist = stats.gamma(a=params["shape"], scale=params["scale"])
    elif family == "beta":
        dist = stats.beta(a=params["alpha"], b=params["beta"])
    elif family == "halfcauchy":
        dist = stats.halfcauchy(scale=params["scale"])
    elif family == "t":
        dist = stats.t(df=params["df"], loc=params["loc"], scale=params["scale"])
    else:
        return None, None

    return dist.cdf(grid), dist.pdf(grid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def w2_distance(cdf_p, cdf_q, grid, n_quantile=500):
    """Wasserstein-2 distance between two distributions given their CDFs.

    W_2 = sqrt(integral (F^{-1}(t) - G^{-1}(t))^2 dt)

    Parameters
    ----------
    cdf_p, cdf_q : array-like
        CDF values evaluated on grid.
    grid : array-like
        Shared x-grid.
    n_quantile : int
        Number of points for quantile function evaluation.

    Returns
    -------
    float
        The W2 distance.
    """
    t_grid = np.linspace(0.001, 0.999, n_quantile)
    q_p = _quantile_function(np.asarray(cdf_p), np.asarray(grid), t_grid)
    q_q = _quantile_function(np.asarray(cdf_q), np.asarray(grid), t_grid)
    integrand = (q_p - q_q) ** 2
    w2_sq = np.trapezoid(integrand, t_grid)
    return float(np.sqrt(max(w2_sq, 0.0)))


def pairwise_w2_matrix(experts, grid):
    """Compute pairwise W2 distance matrix for a list of ExpertPrior objects.

    Parameters
    ----------
    experts : list of ExpertPrior
    grid : array-like

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    grid = np.asarray(grid)
    n = len(experts)
    cdfs = []
    for e in experts:
        cdf, _ = _get_cdf_pdf_grid(e, grid)
        cdfs.append(cdf)

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if cdfs[i] is not None and cdfs[j] is not None:
                d = w2_distance(cdfs[i], cdfs[j], grid)
                mat[i, j] = d
                mat[j, i] = d
    return mat


def wasserstein_barycenter(experts, grid, n_quantile=500):
    """Wasserstein barycenter of expert priors (equal weights).

    The barycenter quantile function is the mean of expert quantile functions.

    Parameters
    ----------
    experts : list of ExpertPrior
    grid : array-like
    n_quantile : int

    Returns
    -------
    dict with keys: barycenter_density, barycenter_grid
    """
    grid = np.asarray(grid)
    t_grid = np.linspace(0.001, 0.999, n_quantile)

    quantile_fns = []
    for e in experts:
        cdf, _ = _get_cdf_pdf_grid(e, grid)
        if cdf is not None:
            qf = _quantile_function(cdf, grid, t_grid)
            quantile_fns.append(qf)

    if len(quantile_fns) == 0:
        return {"barycenter_density": np.zeros_like(grid), "barycenter_grid": grid}

    # Barycenter quantile function = mean of quantile functions
    q_bary = np.mean(quantile_fns, axis=0)

    # Convert to PDF on grid
    pdf = _pdf_from_quantile_fn(q_bary, t_grid, grid)

    return {
        "barycenter_density": pdf,
        "barycenter_grid": grid,
    }


def displacement_interpolation(expert_a, expert_b, grid, n_quantile=500,
                                t_values=None):
    """Displacement interpolation between two experts.

    Q_t = (1-t)*Q_0 + t*Q_1

    Parameters
    ----------
    expert_a, expert_b : ExpertPrior
    grid : array-like
    n_quantile : int
    t_values : list of float, default [0, 0.25, 0.5, 0.75, 1]

    Returns
    -------
    list of dict, each with keys: t, density, grid
    """
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    grid = np.asarray(grid)
    u_grid = np.linspace(0.001, 0.999, n_quantile)

    cdf_a, _ = _get_cdf_pdf_grid(expert_a, grid)
    cdf_b, _ = _get_cdf_pdf_grid(expert_b, grid)

    if cdf_a is None or cdf_b is None:
        return [{"t": t, "density": np.zeros_like(grid), "grid": grid} for t in t_values]

    q_a = _quantile_function(cdf_a, grid, u_grid)
    q_b = _quantile_function(cdf_b, grid, u_grid)

    paths = []
    for t in t_values:
        q_t = (1.0 - t) * q_a + t * q_b
        pdf = _pdf_from_quantile_fn(q_t, u_grid, grid)
        paths.append({"t": t, "density": pdf, "grid": grid})

    return paths


def transport_map(expert_a, expert_b, grid, n_quantile=500):
    """Compute transport map T: T(x) = G^{-1}(F(x)).

    Parameters
    ----------
    expert_a, expert_b : ExpertPrior
    grid : array-like
    n_quantile : int

    Returns
    -------
    dict with keys: x_grid, transport_values
    """
    grid = np.asarray(grid)
    u_grid = np.linspace(0.001, 0.999, n_quantile)

    cdf_a, _ = _get_cdf_pdf_grid(expert_a, grid)
    cdf_b, _ = _get_cdf_pdf_grid(expert_b, grid)

    if cdf_a is None or cdf_b is None:
        return {"x_grid": grid, "transport_values": grid}

    # T(x) = G^{-1}(F(x))
    # F(x) evaluated on grid gives cdf_a
    # G^{-1} is the quantile function of b
    q_b = _quantile_function(cdf_b, grid, u_grid)
    # For each x in grid, compute F_a(x), then G_b^{-1}(F_a(x))
    f_x = np.clip(cdf_a, 0.001, 0.999)
    t_vals = np.interp(f_x, u_grid, q_b)

    return {"x_grid": grid, "transport_values": t_vals}


def optimal_transport_analysis(experts, grid=None, n_quantile=500):
    """Full optimal transport analysis for a set of expert priors.

    Parameters
    ----------
    experts : list of ExpertPrior
        Each must have best_fit set.
    grid : array-like, optional
        Shared evaluation grid. Auto-generated if None.
    n_quantile : int

    Returns
    -------
    dict with keys:
        w2_matrix : np.ndarray
        barycenter_density : np.ndarray
        barycenter_grid : np.ndarray
        displacement_paths : list (for first two experts if >=2)
        transport_maps : list of dict (for first two experts if >=2)
    """
    if grid is None:
        grid = np.linspace(-5, 5, 500)
    grid = np.asarray(grid)

    w2_mat = pairwise_w2_matrix(experts, grid)
    bary = wasserstein_barycenter(experts, grid, n_quantile)

    disp_paths = []
    t_maps = []
    if len(experts) >= 2:
        disp_paths = displacement_interpolation(
            experts[0], experts[1], grid, n_quantile
        )
        tm = transport_map(experts[0], experts[1], grid, n_quantile)
        t_maps.append(tm)

    return {
        "w2_matrix": w2_mat,
        "barycenter_density": bary["barycenter_density"],
        "barycenter_grid": bary["barycenter_grid"],
        "displacement_paths": disp_paths,
        "transport_maps": t_maps,
    }
