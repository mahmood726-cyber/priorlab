import pytest
from priorlab.models import ElicitedQuantiles
from priorlab.fitting import fit_all_distributions, fit_normal, fit_lognormal, fit_gamma, select_best_fit


def test_normal_fit_symmetric(symmetric_quantiles):
    result = fit_normal(symmetric_quantiles)
    assert result.family == "normal"
    assert abs(result.params["mu"] - symmetric_quantiles.median) < 0.1
    assert result.params["sigma"] > 0


def test_lognormal_fit_skewed(skewed_quantiles):
    result = fit_lognormal(skewed_quantiles)
    assert result.family == "lognormal"
    assert result.params["mu_log"] > -10
    assert result.params["sigma_log"] > 0


def test_fit_all_returns_multiple(symmetric_quantiles):
    fits = fit_all_distributions(symmetric_quantiles)
    assert len(fits) >= 4  # normal, t always; gamma/halfcauchy may fail for negative median


def test_ks_distance_nonnegative(symmetric_quantiles):
    fits = fit_all_distributions(symmetric_quantiles)
    for f in fits:
        assert f.ks_distance >= 0.0


def test_best_fit_is_lowest_ks(symmetric_quantiles):
    fits = fit_all_distributions(symmetric_quantiles)
    best = select_best_fit(fits)
    ks_values = [f.ks_distance for f in fits]
    assert best.ks_distance == min(ks_values)


def test_normal_perfect_match():
    """Quantiles generated from N(0, 1) should fit Normal perfectly."""
    from scipy.stats import norm
    q = ElicitedQuantiles(
        lower=norm.ppf(0.05), q1=norm.ppf(0.25), median=0.0,
        q3=norm.ppf(0.75), upper=norm.ppf(0.95),
    )
    result = fit_normal(q)
    assert abs(result.params["mu"]) < 0.05
    assert abs(result.params["sigma"] - 1.0) < 0.1
    assert result.ks_distance < 0.02


def test_pdf_values_populated(symmetric_quantiles):
    fits = fit_all_distributions(symmetric_quantiles)
    successful = [f for f in fits if f.ks_distance < 900]
    assert len(successful) >= 2
    for f in successful:
        assert len(f.pdf_values) > 0
        assert len(f.x_grid) == len(f.pdf_values)


def test_gamma_fit_positive(skewed_quantiles):
    result = fit_gamma(skewed_quantiles)
    assert result.family == "gamma"
    assert result.params["shape"] > 0
    assert result.params["scale"] > 0
