import pytest
from priorlab.models import ElicitedQuantiles, RouletteBins


@pytest.fixture
def symmetric_quantiles():
    """Symmetric (Normal-like) quantiles for treatment effect."""
    return ElicitedQuantiles(lower=-0.8, q1=-0.4, median=-0.2, q3=0.0, upper=0.3)


@pytest.fixture
def skewed_quantiles():
    """Right-skewed quantiles for tau-squared (positive-only)."""
    return ElicitedQuantiles(lower=0.001, q1=0.02, median=0.08, q3=0.20, upper=0.50)


@pytest.fixture
def bounded_quantiles():
    """Bounded [0,1] quantiles for baseline risk."""
    return ElicitedQuantiles(lower=0.02, q1=0.05, median=0.10, q3=0.20, upper=0.40)


@pytest.fixture
def uniform_roulette():
    """Uniform chip distribution (2 chips per bin)."""
    edges = [i * 0.1 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    chips = [2] * 10
    return RouletteBins(bin_edges=edges, chips=chips)


@pytest.fixture
def concentrated_roulette():
    """Chips concentrated in center bins."""
    edges = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    chips = [0, 1, 2, 5, 7, 3, 1, 1, 0, 0]
    return RouletteBins(bin_edges=edges, chips=chips)
