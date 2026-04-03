import numpy as np
import pytest
from scipy.stats import norm

from priorlab.decision_theory import (
    admissibility_check,
    bayes_risk,
    entropic_risk,
    minimax_prior,
    prior_selection_ranking,
    spectral_risk,
)

GRID = np.linspace(-5, 5, 300)


def _make_priors():
    """Three candidate priors with different spreads."""
    return {
        "narrow": norm(loc=0, scale=0.5).pdf(GRID),
        "medium": norm(loc=0, scale=1.0).pdf(GRID),
        "wide": norm(loc=0, scale=2.0).pdf(GRID),
    }


def test_bayes_risk_nonnegative():
    """Bayes risk (posterior variance) must be non-negative."""
    priors = _make_priors()
    for label, pdf in priors.items():
        r = bayes_risk(pdf, GRID, likelihood_sigma=1.0)
        assert r >= 0, f"Negative Bayes risk for {label}: {r}"


def test_admissible_set_nonempty():
    """Admissible set must contain at least one prior."""
    priors = _make_priors()
    result = admissibility_check(priors, GRID)
    assert len(result["admissible"]) >= 1, "Admissible set is empty"


def test_entropic_risk_exceeds_mean():
    """Entropic risk >= E[X] for beta > 0 (Jensen's inequality)."""
    losses = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    er = entropic_risk(losses, beta=1.0)
    assert er >= np.mean(losses) - 1e-10, \
        f"Entropic risk {er} < mean {np.mean(losses)}"


def test_minimax_returns_valid_label():
    """minimax_prior must return one of the candidate labels."""
    priors = _make_priors()
    mm = minimax_prior(priors, GRID)
    assert mm in priors, f"Minimax label '{mm}' not in candidates"


def test_ranking_returns_all_keys():
    """prior_selection_ranking returns all expected keys and sorted ranking."""
    priors = _make_priors()
    result = prior_selection_ranking(priors, GRID, seed=42)
    expected = {
        "bayes_risks", "entropic_risks", "spectral_risks",
        "admissible_priors", "minimax_prior", "ranking",
        "complete_class_size",
    }
    assert expected == set(result.keys())
    # Ranking should be sorted ascending by score
    scores = [r["score"] for r in result["ranking"]]
    assert scores == sorted(scores), "Ranking is not sorted by score"
    assert result["complete_class_size"] >= 1
