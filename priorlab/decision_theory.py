"""Axiomatic Decision Theory for Prior Selection.

Choose between candidate priors using Bayes risk, entropic risk,
spectral risk measures, admissibility checks, minimax criteria,
and composite scoring.
"""

import numpy as np


def bayes_risk(prior_pdf, grid, likelihood_sigma=1.0):
    """Compute Bayes risk under squared error loss.

    For Normal likelihood with known sigma, the Bayes action is the
    posterior mean and Bayes risk = posterior variance.

    We approximate: posterior variance = 1 / (1/prior_var + 1/sigma^2).

    Parameters
    ----------
    prior_pdf : array-like
        Prior density on *grid*.
    grid : array-like
        Evaluation grid.
    likelihood_sigma : float
        Likelihood standard deviation.

    Returns
    -------
    float: Bayes risk (posterior variance).
    """
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(prior_pdf, dtype=float)
    pdf = np.maximum(pdf, 0.0)

    # Normalize
    integral = np.trapezoid(pdf, grid)
    if integral > 0:
        pdf = pdf / integral

    # Prior mean and variance
    prior_mean = float(np.trapezoid(grid * pdf, grid))
    prior_var = float(np.trapezoid((grid - prior_mean) ** 2 * pdf, grid))
    prior_var = max(prior_var, 1e-15)

    # Posterior variance under Normal likelihood
    post_var = 1.0 / (1.0 / prior_var + 1.0 / (likelihood_sigma ** 2))
    return post_var


def entropic_risk(losses, beta=1.0):
    """Entropic risk measure: rho(X) = (1/beta) * log(E[exp(beta*X)]).

    Penalizes tail losses exponentially.

    Parameters
    ----------
    losses : array-like
        Loss values (Monte Carlo samples or grid-based).
    beta : float
        Risk aversion parameter (default 1.0).

    Returns
    -------
    float: entropic risk.
    """
    losses = np.asarray(losses, dtype=float)
    # Numerically stable: subtract max before exp
    max_l = np.max(losses)
    log_mean_exp = max_l + np.log(np.mean(np.exp(beta * (losses - max_l))))
    return float(log_mean_exp / beta)


def spectral_risk(losses, phi_fn=None):
    """Spectral risk measure: integral of VaR(alpha) * phi(alpha) dalpha.

    Parameters
    ----------
    losses : array-like
        Loss values.
    phi_fn : callable or None
        Weight function phi(alpha). Default: phi(alpha) = 2*alpha
        (equivalent to CVaR at 50%).

    Returns
    -------
    float: spectral risk.
    """
    losses = np.sort(np.asarray(losses, dtype=float))
    n = len(losses)

    if phi_fn is None:
        phi_fn = lambda alpha: 2.0 * alpha

    # Approximate: VaR(alpha) = quantile at level alpha
    # Spectral risk = integral_0^1 VaR(alpha) * phi(alpha) dalpha
    alphas = np.linspace(0.001, 0.999, 200)
    var_vals = np.percentile(losses, alphas * 100)
    phi_vals = np.array([phi_fn(a) for a in alphas])

    return float(np.trapezoid(var_vals * phi_vals, alphas))


def _compute_loss_samples(prior_pdf, grid, theta_grid, likelihood_sigma=1.0,
                          n_samples=500, seed=42):
    """Generate squared error loss samples for a prior.

    For each theta in theta_grid, compute L(theta, d*) = (theta - d*)^2
    where d* is the posterior mean.
    """
    rng = np.random.default_rng(seed)
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(prior_pdf, dtype=float)
    pdf = np.maximum(pdf, 0.0)

    integral = np.trapezoid(pdf, grid)
    if integral > 0:
        pdf = pdf / integral

    # Prior mean (Bayes action under squared error)
    prior_mean = float(np.trapezoid(grid * pdf, grid))
    prior_var = float(np.trapezoid((grid - prior_mean) ** 2 * pdf, grid))
    prior_var = max(prior_var, 1e-15)

    sigma2 = likelihood_sigma ** 2
    post_precision = 1.0 / prior_var + 1.0 / sigma2

    losses = []
    for theta in theta_grid:
        # Posterior mean given one observation x ~ N(theta, sigma2)
        # For expectation over theta: d*(prior) = prior_mean weighted by posterior
        # Simplified: average loss over plausible x values
        # Draw x from likelihood
        xs = rng.normal(theta, likelihood_sigma, size=n_samples)
        for x in xs:
            post_mean = (prior_mean / prior_var + x / sigma2) / post_precision
            loss = (theta - post_mean) ** 2
            losses.append(loss)

    return np.array(losses)


def admissibility_check(priors, grid, likelihood_sigma=1.0):
    """Check admissibility of each candidate prior.

    A prior p is admissible if no other prior q dominates it
    (has lower or equal Bayes risk at every decision point d
    with strict inequality somewhere).

    Parameters
    ----------
    priors : dict of {label: array-like}
        Candidate prior densities on *grid*.
    grid : array-like
        Evaluation grid.
    likelihood_sigma : float
        Likelihood standard deviation.

    Returns
    -------
    dict with keys: admissible (list of labels), dominated (list of labels),
    risk_profiles (dict label -> list of risks at decision grid).
    """
    grid = np.asarray(grid, dtype=float)
    labels = list(priors.keys())

    # Evaluate Bayes risk at each of 20 grid points (different likelihood sigmas)
    sigma_grid = np.linspace(0.1, 3.0, 20)
    risk_profiles = {}

    for label in labels:
        risks = []
        for sig in sigma_grid:
            r = bayes_risk(priors[label], grid, likelihood_sigma=sig)
            risks.append(r)
        risk_profiles[label] = risks

    admissible = []
    dominated = []

    for label_i in labels:
        is_dominated = False
        ri = np.array(risk_profiles[label_i])
        for label_j in labels:
            if label_i == label_j:
                continue
            rj = np.array(risk_profiles[label_j])
            # j dominates i if rj <= ri everywhere and rj < ri somewhere
            if np.all(rj <= ri + 1e-15) and np.any(rj < ri - 1e-15):
                is_dominated = True
                break
        if is_dominated:
            dominated.append(label_i)
        else:
            admissible.append(label_i)

    return {
        "admissible": admissible,
        "dominated": dominated,
        "risk_profiles": risk_profiles,
    }


def minimax_prior(priors, grid, likelihood_sigma=1.0):
    """Find the minimax prior: minimizes maximum Bayes risk over theta.

    Parameters
    ----------
    priors : dict of {label: array-like}
        Candidate prior densities on *grid*.
    grid : array-like
        Evaluation grid.
    likelihood_sigma : float
        Likelihood standard deviation.

    Returns
    -------
    str: label of the minimax prior.
    """
    grid = np.asarray(grid, dtype=float)
    labels = list(priors.keys())

    # Evaluate max risk over a theta grid for each prior
    theta_grid = np.linspace(grid[0], grid[-1], 20)

    best_label = labels[0]
    best_max_risk = np.inf

    for label in labels:
        pdf = np.asarray(priors[label], dtype=float)
        pdf = np.maximum(pdf, 0.0)
        integral = np.trapezoid(pdf, grid)
        if integral > 0:
            pdf = pdf / integral

        prior_mean = float(np.trapezoid(grid * pdf, grid))
        prior_var = float(np.trapezoid((grid - prior_mean) ** 2 * pdf, grid))
        prior_var = max(prior_var, 1e-15)
        sigma2 = likelihood_sigma ** 2
        post_precision = 1.0 / prior_var + 1.0 / sigma2

        max_risk = 0.0
        for theta in theta_grid:
            # Expected loss at this theta under squared error
            # E_x[(theta - d*(x))^2] where d*(x) = posterior mean
            # = posterior variance + (theta - E[d*(x)])^2 part
            # For Normal-Normal: post_mean(x) = (prior_mean/prior_var + x/sigma2)/post_prec
            # E_x[post_mean] = (prior_mean/prior_var + theta/sigma2)/post_prec
            expected_d = (prior_mean / prior_var + theta / sigma2) / post_precision
            bias_sq = (theta - expected_d) ** 2
            post_var = 1.0 / post_precision
            risk = post_var + bias_sq
            max_risk = max(max_risk, risk)

        if max_risk < best_max_risk:
            best_max_risk = max_risk
            best_label = label

    return best_label


def prior_selection_ranking(priors, grid, likelihood_sigma=1.0, seed=42):
    """Rank candidate priors using combined decision-theoretic criteria.

    Score = 0.5 * normalized_bayes_risk + 0.3 * admissibility_bonus
            + 0.2 * minimax_bonus

    Lower score is better.

    Parameters
    ----------
    priors : dict of {label: array-like}
        Candidate prior densities on *grid*.
    grid : array-like
        Evaluation grid.
    likelihood_sigma : float
        Likelihood standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: bayes_risks, entropic_risks, spectral_risks,
    admissible_priors, minimax_prior, ranking, complete_class_size.
    """
    grid = np.asarray(grid, dtype=float)
    labels = list(priors.keys())

    # Bayes risks
    b_risks = {}
    for label in labels:
        b_risks[label] = bayes_risk(priors[label], grid, likelihood_sigma)

    # Loss samples for entropic and spectral risks
    theta_grid = np.linspace(grid[0], grid[-1], 10)
    e_risks = {}
    s_risks = {}
    for label in labels:
        losses = _compute_loss_samples(priors[label], grid, theta_grid,
                                        likelihood_sigma, n_samples=100, seed=seed)
        e_risks[label] = entropic_risk(losses, beta=1.0)
        s_risks[label] = spectral_risk(losses)

    # Admissibility
    adm = admissibility_check(priors, grid, likelihood_sigma)
    admissible_labels = adm["admissible"]

    # Minimax
    mm_prior = minimax_prior(priors, grid, likelihood_sigma)

    # Normalize Bayes risks to [0, 1]
    br_vals = np.array([b_risks[l] for l in labels])
    br_min, br_max = br_vals.min(), br_vals.max()
    br_range = br_max - br_min if br_max > br_min else 1.0

    ranking = []
    for label in labels:
        norm_br = (b_risks[label] - br_min) / br_range
        adm_bonus = 0.0 if label in admissible_labels else 1.0
        mm_bonus = 0.0 if label == mm_prior else 1.0
        score = 0.5 * norm_br + 0.3 * adm_bonus + 0.2 * mm_bonus
        ranking.append({"prior": label, "score": round(score, 6)})

    ranking.sort(key=lambda x: x["score"])

    return {
        "bayes_risks": b_risks,
        "entropic_risks": e_risks,
        "spectral_risks": s_risks,
        "admissible_priors": admissible_labels,
        "minimax_prior": mm_prior,
        "ranking": ranking,
        "complete_class_size": len(admissible_labels),
    }
