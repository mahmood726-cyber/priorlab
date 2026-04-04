"""Statistical Depth Functions for Expert Prior Assessment.

Tukey (halfspace), simplicial, and zonoid depth functions for 1D
distributions, with expert ranking, depth contours, and trimmed
aggregation based on depth.
"""

import numpy as np
from scipy import stats
from priorlab.models import ExpertPrior


def _get_cdf_on_grid(expert: ExpertPrior, grid):
    """Return CDF values on grid from expert's best fit distribution."""
    fitted = expert.best_fit
    if fitted is None:
        return None

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
        return None

    return dist.cdf(grid)


def _get_pdf_on_grid(expert: ExpertPrior, grid):
    """Return PDF values on grid from expert's best fit distribution."""
    fitted = expert.best_fit
    if fitted is None:
        return None

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
        return None

    return dist.pdf(grid)


# ---------------------------------------------------------------------------
# Depth functions (1D)
# ---------------------------------------------------------------------------

def tukey_depth(x, cdf_vals):
    """Tukey (halfspace) depth in 1D: D_T(x) = min(F(x), 1 - F(x)).

    Parameters
    ----------
    x : array-like
        Not used directly; depth is computed from CDF values.
    cdf_vals : array-like
        CDF values F(x) at the grid points.

    Returns
    -------
    np.ndarray
        Tukey depth values in [0, 0.5].
    """
    f = np.asarray(cdf_vals, dtype=float)
    f = np.clip(f, 0.0, 1.0)
    return np.minimum(f, 1.0 - f)


def simplicial_depth(x, cdf_vals):
    """Simplicial depth in 1D: D_S(x) = 2 * F(x) * (1 - F(x)).

    Parameters
    ----------
    x : array-like
        Not used directly.
    cdf_vals : array-like
        CDF values F(x) at the grid points.

    Returns
    -------
    np.ndarray
        Simplicial depth values in [0, 0.5].
    """
    f = np.asarray(cdf_vals, dtype=float)
    f = np.clip(f, 0.0, 1.0)
    return 2.0 * f * (1.0 - f)


def zonoid_depth(x, cdf_vals):
    """Zonoid depth in 1D: D_Z(x) = 1 - |F(x) - 0.5| / 0.5.

    Parameters
    ----------
    x : array-like
        Not used directly.
    cdf_vals : array-like
        CDF values F(x) at the grid points.

    Returns
    -------
    np.ndarray
        Zonoid depth values in [0, 1].
    """
    f = np.asarray(cdf_vals, dtype=float)
    f = np.clip(f, 0.0, 1.0)
    return 1.0 - np.abs(f - 0.5) / 0.5


# ---------------------------------------------------------------------------
# Consensus CDF for multi-expert depth calculations
# ---------------------------------------------------------------------------

def _consensus_cdf(experts, grid):
    """Simple linear pool CDF: mean of expert CDFs."""
    cdfs = []
    for e in experts:
        cdf = _get_cdf_on_grid(e, grid)
        if cdf is not None:
            cdfs.append(cdf)
    if len(cdfs) == 0:
        return np.full_like(grid, 0.5)
    return np.mean(cdfs, axis=0)


# ---------------------------------------------------------------------------
# Expert depth analysis
# ---------------------------------------------------------------------------

def expert_depths(experts, grid):
    """Compute depth of each expert's median in the consensus distribution.

    Parameters
    ----------
    experts : list of ExpertPrior
    grid : array-like

    Returns
    -------
    dict mapping expert_id -> float (Tukey depth of median in consensus)
    """
    grid = np.asarray(grid)
    cons_cdf = _consensus_cdf(experts, grid)

    result = {}
    for e in experts:
        if e.quantiles is not None:
            median_val = e.quantiles.median
        elif e.best_fit is not None and "mu" in e.best_fit.params:
            median_val = e.best_fit.params["mu"]
        elif e.best_fit is not None and "loc" in e.best_fit.params:
            median_val = e.best_fit.params["loc"]
        else:
            result[e.expert_id] = 0.0
            continue

        # CDF at median in consensus distribution
        f_median = float(np.interp(median_val, grid, cons_cdf))
        depth = min(f_median, 1.0 - f_median)
        result[e.expert_id] = depth

    return result


def depth_ranking(experts, grid):
    """Rank experts from most central to most extreme.

    Parameters
    ----------
    experts : list of ExpertPrior
    grid : array-like

    Returns
    -------
    list of (expert_id, depth) tuples, sorted descending by depth.
    """
    depths = expert_depths(experts, grid)
    return sorted(depths.items(), key=lambda x: x[1], reverse=True)


def depth_contours(grid, cdf_vals, alphas=None):
    """Compute depth contours: {x : D_T(x) >= alpha} for each alpha.

    Parameters
    ----------
    grid : array-like
    cdf_vals : array-like
        CDF values on grid (consensus or single distribution).
    alphas : list of float
        Depth levels. Default [0.1, 0.25, 0.5].

    Returns
    -------
    dict mapping alpha -> (x_min, x_max) contour bounds.
    """
    if alphas is None:
        alphas = [0.1, 0.25, 0.5]

    grid = np.asarray(grid)
    td = tukey_depth(grid, cdf_vals)

    contours = {}
    for alpha in alphas:
        mask = td >= alpha
        if np.any(mask):
            indices = np.where(mask)[0]
            contours[alpha] = (float(grid[indices[0]]), float(grid[indices[-1]]))
        else:
            contours[alpha] = None
    return contours


def trimmed_aggregation(experts, grid, trim_fraction=None):
    """Remove the shallowest expert and re-aggregate (linear pool).

    Parameters
    ----------
    experts : list of ExpertPrior
    grid : array-like
    trim_fraction : float, optional
        Not used; always removes exactly 1 expert (the shallowest).

    Returns
    -------
    dict with keys:
        removed_expert : str (expert_id of removed expert)
        trimmed_density : np.ndarray
        trimmed_grid : np.ndarray
    """
    grid = np.asarray(grid)

    if len(experts) <= 1:
        pdfs = []
        for e in experts:
            pdf = _get_pdf_on_grid(e, grid)
            if pdf is not None:
                pdfs.append(pdf)
        agg = np.mean(pdfs, axis=0) if pdfs else np.zeros_like(grid)
        return {
            "removed_expert": None,
            "trimmed_density": agg,
            "trimmed_grid": grid,
        }

    ranking = depth_ranking(experts, grid)
    # Shallowest expert is last in ranking (lowest depth)
    shallowest_id = ranking[-1][0]

    # Re-aggregate without the shallowest
    remaining_pdfs = []
    for e in experts:
        if e.expert_id == shallowest_id:
            continue
        pdf = _get_pdf_on_grid(e, grid)
        if pdf is not None:
            remaining_pdfs.append(pdf)

    if len(remaining_pdfs) == 0:
        trimmed = np.zeros_like(grid)
    else:
        trimmed = np.mean(remaining_pdfs, axis=0)
        # Normalize
        area = np.trapezoid(trimmed, grid)
        if area > 1e-15:
            trimmed = trimmed / area

    return {
        "removed_expert": shallowest_id,
        "trimmed_density": trimmed,
        "trimmed_grid": grid,
    }


def depth_analysis(experts, grid=None, alphas=None):
    """Full depth analysis for a set of expert priors.

    Parameters
    ----------
    experts : list of ExpertPrior
    grid : array-like, optional
    alphas : list of float, optional

    Returns
    -------
    dict with keys:
        tukey_depth_grid : np.ndarray
        simplicial_depth_grid : np.ndarray
        expert_depths : dict
        depth_ranking : list of (expert_id, depth) tuples
        depth_contours : dict
        trimmed_aggregation : dict
    """
    if grid is None:
        grid = np.linspace(-5, 5, 500)
    grid = np.asarray(grid)

    cons_cdf = _consensus_cdf(experts, grid)
    td = tukey_depth(grid, cons_cdf)
    sd = simplicial_depth(grid, cons_cdf)

    return {
        "tukey_depth_grid": td,
        "simplicial_depth_grid": sd,
        "expert_depths": expert_depths(experts, grid),
        "depth_ranking": depth_ranking(experts, grid),
        "depth_contours": depth_contours(grid, cons_cdf, alphas),
        "trimmed_aggregation": trimmed_aggregation(experts, grid),
    }
