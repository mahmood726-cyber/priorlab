import numpy as np
import pytest
from scipy.stats import norm

from priorlab.models import ElicitedQuantiles, FittedDistribution
from priorlab.calibration import (
    calibrate_prior,
    calibration_score,
    hit_rate_test,
    pit_test,
    sharpness,
)


@pytest.fixture
def well_specified():
    """A Normal(0,1) prior where quantiles come from the same distribution."""
    q = ElicitedQuantiles(
        lower=norm.ppf(0.05),
        q1=norm.ppf(0.25),
        median=0.0,
        q3=norm.ppf(0.75),
        upper=norm.ppf(0.95),
    )
    fitted = FittedDistribution(
        family="normal",
        params={"mu": 0.0, "sigma": 1.0},
    )
    return q, fitted


def test_hit_rates_near_nominal(well_specified):
    """For a well-specified prior, hit rates should be near nominal levels."""
    q, fitted = well_specified
    hr = hit_rate_test(q, fitted, n_sim=5000, seed=42)
    for level in [0.05, 0.25, 0.50, 0.75, 0.95]:
        assert abs(hr[level] - level) < 0.05, f"Level {level}: got {hr[level]}"


def test_pit_uniform_for_correct_prior(well_specified):
    """PIT of well-calibrated prior should pass KS test for uniformity."""
    _, fitted = well_specified
    result = pit_test(fitted, n_sim=5000, seed=42)
    assert result["p_value"] > 0.05  # should NOT reject uniformity
    assert 0 <= result["ks_stat"] <= 1


def test_calibration_score_low_for_correct_prior(well_specified):
    """Calibration score should be near zero for well-specified prior."""
    q, fitted = well_specified
    score = calibration_score(q, fitted, n_sim=5000, seed=42)
    assert score < 0.005  # Brier-like score near zero


def test_sharpness_positive():
    """Sharpness (90% CI width) should always be positive."""
    q = ElicitedQuantiles(lower=-1.645, q1=-0.674, median=0.0, q3=0.674, upper=1.645)
    s = sharpness(q)
    assert s > 0
    assert abs(s - (1.645 * 2)) < 1e-6


def test_calibrate_prior_full(well_specified):
    """Full calibration returns all keys and is_calibrated=True for correct prior."""
    q, fitted = well_specified
    result = calibrate_prior(q, fitted, n_sim=5000, seed=42)
    expected_keys = {
        "hit_rates", "pit_ks_stat", "pit_p_value",
        "calibration_score", "sharpness", "is_calibrated",
    }
    assert set(result.keys()) == expected_keys
    assert result["is_calibrated"] is True
    assert result["calibration_score"] < 0.005
    assert result["sharpness"] > 0
