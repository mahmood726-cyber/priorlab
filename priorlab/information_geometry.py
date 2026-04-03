"""Fisher-Rao metric on the prior manifold.

Riemannian geometry of the space of Normal prior distributions, including
the Fisher information metric, geodesic distances, natural gradients,
geodesic paths, and scalar curvature.
"""

import numpy as np


def fisher_metric(sigma):
    """Fisher information metric for Normal(mu, sigma^2) at given sigma.

    The metric tensor g for the Normal family parameterized by (mu, sigma) is:
        g_11 = 1/sigma^2   (mu direction)
        g_22 = 2/sigma^2   (sigma direction)
        g_12 = g_21 = 0
        det(g) = 2/sigma^4

    Parameters
    ----------
    sigma : float
        Standard deviation (> 0).

    Returns
    -------
    dict with keys: metric (2x2 list), determinant (float).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    s2 = sigma ** 2
    g = [[1.0 / s2, 0.0],
         [0.0, 2.0 / s2]]
    det_g = 2.0 / (s2 ** 2)
    return {
        "metric": g,
        "determinant": det_g,
    }


def fisher_rao_distance(mu1, sigma1, mu2, sigma2):
    """Exact Fisher-Rao geodesic distance between two univariate Normals.

    d_FR(N(mu1,s1^2), N(mu2,s2^2)) =
        sqrt(2) * arccosh(1 + ((mu1-mu2)^2 + 2*(s1-s2)^2) / (2*s1*s2))

    Parameters
    ----------
    mu1, sigma1 : float
        Parameters of first Normal.
    mu2, sigma2 : float
        Parameters of second Normal.

    Returns
    -------
    float
        Geodesic distance (>= 0).
    """
    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("sigma values must be positive")
    delta_mu = mu1 - mu2
    delta_s = sigma1 - sigma2
    argument = 1.0 + (delta_mu ** 2 + 2.0 * delta_s ** 2) / (2.0 * sigma1 * sigma2)
    # arccosh is defined for argument >= 1
    argument = max(argument, 1.0)
    return float(np.sqrt(2.0) * np.arccosh(argument))


def pairwise_geodesic_matrix(experts):
    """Pairwise Fisher-Rao geodesic distance matrix for multiple experts.

    Parameters
    ----------
    experts : list of dict
        Each dict has keys 'mu' and 'sigma' (best-fit Normal parameters).

    Returns
    -------
    numpy.ndarray
        n x n symmetric distance matrix.
    """
    n = len(experts)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = fisher_rao_distance(
                experts[i]["mu"], experts[i]["sigma"],
                experts[j]["mu"], experts[j]["sigma"],
            )
            mat[i, j] = d
            mat[j, i] = d
    return mat


def natural_gradient(mu, sigma, target_mu, target_sigma):
    """Natural gradient: F^{-1} * Euclidean gradient toward target.

    For the Normal family with Fisher metric F:
        F = diag(1/sigma^2, 2/sigma^2)
        F^{-1} = diag(sigma^2, sigma^2/2)

    The Euclidean gradient of KL toward target is:
        grad_mu = -(target_mu - mu) / sigma^2 (but we want the direction)
        grad_sigma = ...

    For simplicity, we compute the natural gradient as the direction of
    steepest descent in the Riemannian sense:
        natural_grad_mu = sigma^2 * (target_mu - mu)
        natural_grad_sigma = (sigma^2 / 2) * (target_sigma^2 / sigma^2 - 1)

    Parameters
    ----------
    mu, sigma : float
        Current prior parameters.
    target_mu, target_sigma : float
        Target parameters (e.g., posterior).

    Returns
    -------
    dict with keys: natural_grad_mu, natural_grad_sigma.
    """
    if sigma <= 0 or target_sigma <= 0:
        raise ValueError("sigma values must be positive")
    s2 = sigma ** 2
    ng_mu = s2 * (target_mu - mu)
    ng_sigma = (s2 / 2.0) * (target_sigma ** 2 / s2 - 1.0)
    return {
        "natural_grad_mu": float(ng_mu),
        "natural_grad_sigma": float(ng_sigma),
    }


def geodesic_path(mu_a, sigma_a, mu_b, sigma_b, n_steps=20):
    """Geodesic path between two Normals via natural parameter interpolation.

    The natural parameters of Normal are:
        eta1 = mu / sigma^2
        eta2 = -1 / (2 * sigma^2)

    The geodesic is approximated by linear interpolation in natural parameter
    space: eta(t) = (1-t)*eta_A + t*eta_B for t in [0, 1].

    Parameters
    ----------
    mu_a, sigma_a : float
        Starting Normal parameters.
    mu_b, sigma_b : float
        Ending Normal parameters.
    n_steps : int
        Number of steps along the path (including endpoints).

    Returns
    -------
    list of dict
        Each dict has keys: t, mu, sigma.
    """
    if sigma_a <= 0 or sigma_b <= 0:
        raise ValueError("sigma values must be positive")

    # Natural parameters
    eta1_a = mu_a / (sigma_a ** 2)
    eta2_a = -1.0 / (2.0 * sigma_a ** 2)
    eta1_b = mu_b / (sigma_b ** 2)
    eta2_b = -1.0 / (2.0 * sigma_b ** 2)

    path = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        eta1 = (1.0 - t) * eta1_a + t * eta1_b
        eta2 = (1.0 - t) * eta2_a + t * eta2_b

        # Convert back: sigma^2 = -1/(2*eta2), mu = eta1 * sigma^2
        if eta2 >= 0:
            # Degenerate case, skip
            continue
        sigma2 = -1.0 / (2.0 * eta2)
        mu = eta1 * sigma2
        sigma = np.sqrt(sigma2)
        path.append({
            "t": round(t, 6),
            "mu": round(float(mu), 6),
            "sigma": round(float(sigma), 6),
        })

    return path


def scalar_curvature():
    """Scalar curvature of the Normal statistical manifold.

    For the univariate Normal family, the scalar curvature under the
    Fisher-Rao metric is a constant: R = -1/2 (hyperbolic geometry).

    Returns
    -------
    float
        -0.5
    """
    return -0.5


def information_geometry(experts, target_mu=None, target_sigma=None):
    """Full information geometry analysis for a panel of Normal experts.

    Parameters
    ----------
    experts : list of dict
        Each dict has keys 'mu' and 'sigma'.
    target_mu : float or None
        Target mu for natural gradient (e.g., posterior mean).
    target_sigma : float or None
        Target sigma for natural gradient.

    Returns
    -------
    dict with keys: geodesic_distances, natural_gradient, geodesic_path,
    scalar_curvature, fisher_metric.
    """
    # Pairwise distance matrix
    dist_mat = pairwise_geodesic_matrix(experts)

    # Fisher metric at the first expert (representative)
    if len(experts) > 0:
        fm = fisher_metric(experts[0]["sigma"])
    else:
        fm = {"metric": [[0, 0], [0, 0]], "determinant": 0}

    # Natural gradient (first expert toward target)
    ng = {"natural_grad_mu": 0.0, "natural_grad_sigma": 0.0}
    if target_mu is not None and target_sigma is not None and len(experts) > 0:
        ng = natural_gradient(
            experts[0]["mu"], experts[0]["sigma"],
            target_mu, target_sigma,
        )

    # Geodesic path between first and last expert
    gp = []
    if len(experts) >= 2:
        gp = geodesic_path(
            experts[0]["mu"], experts[0]["sigma"],
            experts[-1]["mu"], experts[-1]["sigma"],
        )

    return {
        "geodesic_distances": dist_mat.tolist(),
        "natural_gradient": ng,
        "geodesic_path": gp,
        "scalar_curvature": scalar_curvature(),
        "fisher_metric": fm["metric"],
    }
