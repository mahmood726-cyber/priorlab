import pytest
from priorlab.models import ElicitedQuantiles
from priorlab.optimal_elicitation import (
    current_entropy,
    elicitation_efficiency,
    optimal_elicitation,
    optimal_next_question,
    sequential_design,
)


@pytest.fixture
def base_quantiles():
    """Standard 5-quantile elicitation."""
    return ElicitedQuantiles(lower=-2.0, q1=-0.5, median=0.0, q3=0.5, upper=2.0)


def test_entropy_is_nonnegative(base_quantiles):
    """Differential entropy of a fitted distribution should be non-negative for broad priors."""
    h = current_entropy(base_quantiles)
    # For a wide Normal-like distribution, entropy is positive
    assert h > 0


def test_optimal_question_not_already_asked(base_quantiles):
    """Optimal next question must not be one of the standard 5 quantiles."""
    result = optimal_next_question(base_quantiles, already_asked=[], n_simulations=20)
    standard = {0.05, 0.25, 0.50, 0.75, 0.95}
    if result["optimal_next_p"] is not None:
        assert result["optimal_next_p"] not in standard


def test_sequential_design_returns_list(base_quantiles):
    """Sequential design should return a list of quantile levels."""
    seq = sequential_design(base_quantiles, k=2, n_simulations=20)
    assert isinstance(seq, list)
    assert len(seq) <= 2
    # All values should be valid probabilities
    for p in seq:
        assert 0 < p < 1


def test_efficiency_positive(base_quantiles):
    """Elicitation efficiency should be positive for asked questions."""
    eff = elicitation_efficiency(base_quantiles, n_questions_asked=5)
    assert eff > 0


def test_full_optimal_elicitation(base_quantiles):
    """Full optimal_elicitation returns all expected keys."""
    result = optimal_elicitation(base_quantiles, k=2, n_simulations=20)
    expected_keys = [
        "optimal_next_p", "expected_gain_by_p", "sequential_design",
        "efficiency_curve", "should_stop", "current_entropy", "questions_asked",
    ]
    for key in expected_keys:
        assert key in result
    assert result["current_entropy"] > 0
    assert result["questions_asked"] >= 5
    assert isinstance(result["should_stop"], bool)
