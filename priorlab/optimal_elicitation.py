"""Optimal question design for prior elicitation.

Information-theoretic sequential design: determines which quantile to ask next
to maximally reduce uncertainty about the expert's prior distribution.
"""

import numpy as np

from priorlab.models import ElicitedQuantiles
from priorlab.fitting import fit_all_distributions, select_best_fit, _make_grid


# Standard quantile levels to consider asking
CANDIDATE_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _entropy_on_grid(pdf, grid, epsilon=1e-300):
    """Differential entropy: H = -integral(p * log(p)) dx."""
    p = np.maximum(np.asarray(pdf, dtype=float), epsilon)
    integrand = -p * np.log(p)
    return float(np.trapezoid(integrand, grid))


def _build_quantiles_from_dict(qdict, lower_p=0.05, upper_p=0.95):
    """Build ElicitedQuantiles from a dict of {p: value} pairs.

    Requires at least the 5 standard quantiles (lower_p, 0.25, 0.50, 0.75, upper_p).
    """
    needed = [lower_p, 0.25, 0.50, 0.75, upper_p]
    for p in needed:
        if p not in qdict:
            return None
    return ElicitedQuantiles(
        lower=qdict[lower_p],
        q1=qdict[0.25],
        median=qdict[0.50],
        q3=qdict[0.75],
        upper=qdict[upper_p],
        lower_p=lower_p,
        upper_p=upper_p,
    )


def current_entropy(quantiles):
    """Compute entropy of current best-fit distribution.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Currently elicited quantiles.

    Returns
    -------
    float
        Differential entropy in nats.
    """
    fits = fit_all_distributions(quantiles)
    best = select_best_fit(fits)
    if not best.pdf_values or not best.x_grid:
        return 0.0
    return _entropy_on_grid(best.pdf_values, best.x_grid)


def _simulate_answer(fitted_dist, p_level, grid):
    """Simulate an expert answer for quantile level p from fitted distribution.

    Uses the inverse CDF approach: find x such that CDF(x) ~ p.
    We approximate by interpolating the CDF on the grid.
    """
    pdf = np.asarray(fitted_dist.pdf_values, dtype=float)
    x = np.asarray(fitted_dist.x_grid, dtype=float)
    # Compute CDF by cumulative trapezoid
    from scipy.integrate import cumulative_trapezoid
    cdf = np.concatenate(([0.0], cumulative_trapezoid(pdf, x)))
    # Interpolate to find quantile
    # Ensure CDF is monotonically non-decreasing
    cdf = np.maximum.accumulate(cdf)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    # Find x at which CDF = p_level
    idx = np.searchsorted(cdf, p_level)
    idx = min(idx, len(x) - 1)
    return float(x[idx])


def expected_information_gain(quantiles, p_level, n_simulations=100):
    """Expected information gain from asking quantile at level p.

    Simulate M answers from the current fitted distribution, refit with
    the additional constraint, and compute the average entropy reduction.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Currently elicited quantiles.
    p_level : float
        Quantile level to evaluate (e.g., 0.10).
    n_simulations : int
        Number of Monte Carlo simulations.

    Returns
    -------
    float
        Expected information gain in nats (>= 0).
    """
    fits = fit_all_distributions(quantiles)
    best = select_best_fit(fits)
    if not best.pdf_values or not best.x_grid:
        return 0.0

    grid = np.asarray(best.x_grid, dtype=float)
    h_current = _entropy_on_grid(best.pdf_values, grid)

    # Build current quantile dict
    qdict = {
        quantiles.lower_p: quantiles.lower,
        0.25: quantiles.q1,
        0.50: quantiles.median,
        0.75: quantiles.q3,
        quantiles.upper_p: quantiles.upper,
    }

    total_reduction = 0.0
    valid_count = 0

    for _ in range(n_simulations):
        # Simulate an answer at p_level
        simulated_value = _simulate_answer(best, p_level, grid)
        # Add small noise to avoid exact duplicates
        noise = np.random.normal(0, 0.001 * (grid[-1] - grid[0]))
        simulated_value += noise

        # Create augmented quantile set
        augmented = dict(qdict)
        augmented[p_level] = simulated_value

        # Try to build new quantiles and refit
        new_eq = _build_quantiles_from_dict(augmented, quantiles.lower_p, quantiles.upper_p)
        if new_eq is None:
            # Can't form a full quantile set - use perturbation approach
            # Perturb the closest standard quantile
            new_eq = _perturb_quantiles(quantiles, p_level, simulated_value)

        new_fits = fit_all_distributions(new_eq)
        new_best = select_best_fit(new_fits)
        if new_best.pdf_values and new_best.x_grid:
            new_grid = np.asarray(new_best.x_grid, dtype=float)
            h_new = _entropy_on_grid(new_best.pdf_values, new_grid)
            total_reduction += max(h_current - h_new, 0.0)
            valid_count += 1

    if valid_count == 0:
        return 0.0
    return total_reduction / valid_count


def _perturb_quantiles(quantiles, p_level, value):
    """Create perturbed quantiles incorporating information from a new quantile level.

    Maps the new quantile level to the closest standard quantile and adjusts.
    """
    standard = {
        0.05: "lower", 0.25: "q1", 0.50: "median", 0.75: "q3", 0.95: "upper"
    }
    # Find closest standard level
    levels = list(standard.keys())
    closest = min(levels, key=lambda x: abs(x - p_level))

    # Create new quantiles with the closest adjusted
    kwargs = {
        "lower": quantiles.lower,
        "q1": quantiles.q1,
        "median": quantiles.median,
        "q3": quantiles.q3,
        "upper": quantiles.upper,
        "lower_p": quantiles.lower_p,
        "upper_p": quantiles.upper_p,
    }

    # Blend the new value with the existing one at the closest level
    old_val = kwargs[standard[closest]]
    weight = 1.0 - abs(p_level - closest)  # higher weight if closer
    weight = max(min(weight, 0.8), 0.2)
    kwargs[standard[closest]] = weight * value + (1 - weight) * old_val

    # Ensure monotonicity
    vals = [kwargs["lower"], kwargs["q1"], kwargs["median"], kwargs["q3"], kwargs["upper"]]
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = vals[i - 1] + 1e-6
    kwargs["lower"], kwargs["q1"], kwargs["median"], kwargs["q3"], kwargs["upper"] = vals

    return ElicitedQuantiles(**kwargs)


def optimal_next_question(quantiles, already_asked=None, n_simulations=50):
    """Find the quantile level with maximum expected information gain.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Currently elicited quantiles.
    already_asked : list of float or None
        Quantile levels already asked (will be excluded from candidates).

    Returns
    -------
    dict with keys: optimal_next_p, expected_gain_by_p.
    """
    if already_asked is None:
        already_asked = []

    # Standard quantiles already used
    standard_asked = {quantiles.lower_p, 0.25, 0.50, 0.75, quantiles.upper_p}
    all_asked = standard_asked | set(already_asked)

    candidates = [p for p in CANDIDATE_LEVELS if p not in all_asked]
    if not candidates:
        return {
            "optimal_next_p": None,
            "expected_gain_by_p": {},
        }

    gain_by_p = {}
    for p in candidates:
        gain = expected_information_gain(quantiles, p, n_simulations=n_simulations)
        gain_by_p[p] = gain

    best_p = max(gain_by_p, key=gain_by_p.get)
    return {
        "optimal_next_p": best_p,
        "expected_gain_by_p": gain_by_p,
    }


def sequential_design(quantiles, k=3, already_asked=None, n_simulations=50):
    """Greedily select the next K quantile questions.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Currently elicited quantiles.
    k : int
        Number of questions to plan.
    already_asked : list of float or None
        Already-asked quantile levels.
    n_simulations : int
        Monte Carlo simulations per candidate.

    Returns
    -------
    list of float
        Ordered list of recommended quantile levels to ask.
    """
    if already_asked is None:
        already_asked = []
    asked = list(already_asked)
    design = []

    for _ in range(k):
        result = optimal_next_question(quantiles, already_asked=asked,
                                       n_simulations=n_simulations)
        if result["optimal_next_p"] is None:
            break
        design.append(result["optimal_next_p"])
        asked.append(result["optimal_next_p"])

    return design


def elicitation_efficiency(quantiles, n_questions_asked):
    """Information gained per question asked.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Current quantiles.
    n_questions_asked : int
        Number of questions asked so far.

    Returns
    -------
    float
        Entropy per question (nats/question). Higher is better.
    """
    if n_questions_asked <= 0:
        return 0.0
    h = current_entropy(quantiles)
    # Entropy is the total information captured; efficiency = H / n
    return h / n_questions_asked


def should_stop(quantiles, already_asked=None, threshold=0.01, n_simulations=50):
    """Stopping rule: stop when expected gain of next question < threshold.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Current quantiles.
    already_asked : list of float or None
        Already-asked quantile levels.
    threshold : float
        Minimum expected gain to continue (in nats).
    n_simulations : int
        Monte Carlo simulations.

    Returns
    -------
    bool
        True if elicitation should stop.
    """
    result = optimal_next_question(quantiles, already_asked=already_asked,
                                   n_simulations=n_simulations)
    if result["optimal_next_p"] is None:
        return True
    max_gain = max(result["expected_gain_by_p"].values()) if result["expected_gain_by_p"] else 0.0
    return max_gain < threshold


def optimal_elicitation(quantiles, already_asked=None, k=3, threshold=0.01,
                        n_simulations=50):
    """Full optimal elicitation analysis.

    Parameters
    ----------
    quantiles : ElicitedQuantiles
        Currently elicited quantiles.
    already_asked : list of float or None
        Quantile levels already asked.
    k : int
        Number of sequential questions to plan.
    threshold : float
        Stopping threshold in nats.
    n_simulations : int
        Monte Carlo simulations per candidate.

    Returns
    -------
    dict with keys: optimal_next_p, expected_gain_by_p, sequential_design,
    efficiency_curve, should_stop, current_entropy, questions_asked.
    """
    if already_asked is None:
        already_asked = []

    # Standard 5 quantiles always asked
    n_standard = 5
    n_questions = n_standard + len(already_asked)

    # Current state
    h_current = current_entropy(quantiles)

    # Optimal next question
    next_q = optimal_next_question(quantiles, already_asked=already_asked,
                                   n_simulations=n_simulations)

    # Sequential design
    seq = sequential_design(quantiles, k=k, already_asked=already_asked,
                            n_simulations=n_simulations)

    # Efficiency curve: cumulative efficiency at each question count
    efficiency_curve = []
    for n in range(1, n_questions + 1):
        efficiency_curve.append({
            "n_questions": n,
            "efficiency": h_current / n,
        })

    # Stopping rule
    stop = False
    if next_q["expected_gain_by_p"]:
        max_gain = max(next_q["expected_gain_by_p"].values())
        stop = max_gain < threshold
    else:
        stop = True

    return {
        "optimal_next_p": next_q["optimal_next_p"],
        "expected_gain_by_p": next_q["expected_gain_by_p"],
        "sequential_design": seq,
        "efficiency_curve": efficiency_curve,
        "should_stop": stop,
        "current_entropy": h_current,
        "questions_asked": n_questions,
    }
