import numpy as np
import pytest

from priorlab.models import ElicitedQuantiles
from priorlab.bootstrap_prior import (
    bayesian_bootstrap_density,
    concentration_rate,
    full_bootstrap_analysis,
)


@pytest.fixture
def sym_q():
    return ElicitedQuantiles(lower=-0.8, q1=-0.4, median=-0.2, q3=0.0, upper=0.3)


def test_density_integrates_to_one(sym_q):
    """Nonparametric bootstrap density should integrate to ~1."""
    result = bayesian_bootstrap_density(sym_q, n_bootstrap=200)
    grid = np.array(result["grid"])
    density = np.array(result["nonparametric_density"])
    integral = np.trapezoid(density, grid)
    assert abs(integral - 1.0) < 0.1, f"Integral = {integral}, expected ~1.0"


def test_credible_band_contains_mean(sym_q):
    """The mean density should lie within credible bands at most grid points."""
    result = bayesian_bootstrap_density(sym_q, n_bootstrap=200)
    density = np.array(result["nonparametric_density"])
    lower = np.array(result["credible_lower"])
    upper = np.array(result["credible_upper"])
    inside = np.sum((density >= lower - 1e-10) & (density <= upper + 1e-10))
    proportion = inside / len(density)
    assert proportion > 0.8, f"Only {proportion:.0%} of mean inside bands"


def test_concentration_rate_decreasing(sym_q):
    """Frechet variance should generally decrease as n_quantiles increases."""
    rates = concentration_rate(sym_q, seed=42)
    # At least the last variance should be smaller than the first
    assert rates[-1]["variance"] < rates[0]["variance"], \
        f"Variance did not decrease: {rates[0]['variance']:.6f} -> {rates[-1]['variance']:.6f}"


def test_bandwidth_positive(sym_q):
    """Bandwidth must be positive."""
    result = bayesian_bootstrap_density(sym_q)
    assert result["bandwidth"] > 0


def test_full_analysis_keys(sym_q):
    """full_bootstrap_analysis returns all expected keys."""
    result = full_bootstrap_analysis(sym_q, n_bootstrap=100)
    expected = {
        "nonparametric_density", "credible_lower", "credible_upper",
        "grid", "concentration_rate", "kl_vs_parametric", "bandwidth",
    }
    assert expected == set(result.keys())
