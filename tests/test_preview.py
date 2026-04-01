import pytest
import numpy as np
from priorlab.preview import conjugate_normal_update, grid_posterior


def test_conjugate_posterior_tighter_than_prior():
    result = conjugate_normal_update(mu_prior=0, sigma_prior=1.0,
                                      theta_data=0.5, se_data=0.5)
    assert result["sigma_post"] < 1.0


def test_strong_prior_dominates():
    result = conjugate_normal_update(mu_prior=0, sigma_prior=0.01,
                                      theta_data=1.0, se_data=1.0)
    assert abs(result["mu_post"]) < 0.1  # close to prior


def test_strong_data_dominates():
    result = conjugate_normal_update(mu_prior=0, sigma_prior=10.0,
                                      theta_data=1.0, se_data=0.01)
    assert abs(result["mu_post"] - 1.0) < 0.1  # close to data


def test_grid_posterior_sums_to_one():
    from scipy.stats import norm
    grid = np.linspace(-3, 3, 500)
    prior_pdf = norm(0, 1).pdf(grid)
    result = grid_posterior(grid, prior_pdf, theta_data=0.5, se_data=0.5)
    integral = np.trapezoid(result, grid)
    assert abs(integral - 1.0) < 0.05
