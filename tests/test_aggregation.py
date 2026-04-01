import numpy as np
import pytest
from priorlab.aggregation import linear_pool, log_pool


def _make_normal_pdf(mu, sigma, grid):
    from scipy.stats import norm
    return norm(loc=mu, scale=sigma).pdf(grid)


def test_linear_pool_identical_priors():
    grid = np.linspace(-2, 2, 200)
    pdf1 = _make_normal_pdf(0, 1, grid)
    pdf2 = _make_normal_pdf(0, 1, grid)
    result = linear_pool([pdf1, pdf2], grid)
    assert np.allclose(result, pdf1, atol=1e-10)


def test_log_pool_identical_priors():
    grid = np.linspace(-2, 2, 200)
    pdf1 = _make_normal_pdf(0, 1, grid)
    result = log_pool([pdf1, pdf1], grid)
    # Normalized log pool of identical = same shape
    ratio = result / pdf1
    assert np.std(ratio[pdf1 > 0.001]) < 0.01  # nearly constant ratio


def test_linear_pool_integrates_to_one():
    grid = np.linspace(-3, 3, 500)
    pdf1 = _make_normal_pdf(-0.5, 0.5, grid)
    pdf2 = _make_normal_pdf(0.5, 0.5, grid)
    result = linear_pool([pdf1, pdf2], grid)
    integral = np.trapezoid(result, grid)
    assert abs(integral - 1.0) < 0.05


def test_equal_weights_default():
    grid = np.linspace(-2, 2, 200)
    pdf1 = _make_normal_pdf(0, 1, grid)
    pdf2 = _make_normal_pdf(0, 1, grid)
    result = linear_pool([pdf1, pdf2], grid, weights=[0.5, 0.5])
    assert np.allclose(result, pdf1, atol=1e-10)
