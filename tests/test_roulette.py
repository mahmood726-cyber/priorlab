import pytest
from priorlab.models import RouletteBins
from priorlab.roulette import roulette_to_quantiles, roulette_to_histogram


def test_chip_normalization(uniform_roulette):
    hist = roulette_to_histogram(uniform_roulette)
    assert abs(sum(hist["probabilities"]) - 1.0) < 0.01


def test_uniform_chips_wide_prior(uniform_roulette):
    quantiles = roulette_to_quantiles(uniform_roulette)
    iqr = quantiles.q3 - quantiles.q1
    total_range = quantiles.upper - quantiles.lower
    assert iqr / total_range > 0.3  # wide spread


def test_concentrated_chips_narrow(concentrated_roulette):
    quantiles = roulette_to_quantiles(concentrated_roulette)
    iqr = quantiles.q3 - quantiles.q1
    assert iqr < 0.8  # narrower than range


def test_empty_bins_zero_prob():
    bins = RouletteBins(bin_edges=[0.0, 0.5, 1.0], chips=[0, 10])
    hist = roulette_to_histogram(bins)
    assert hist["probabilities"][0] == 0.0
    assert hist["probabilities"][1] == 1.0
