"""Tests for Gaussian copula joint prior elicitation."""

import numpy as np
import pytest
from scipy import stats

from priorlab.copula import (
    gaussian_copula,
    joint_density,
    kendall_tau,
    conditional_pdf,
)


def test_joint_density_integrates_to_one():
    """2D joint density should integrate to approximately 1."""
    grid1 = np.linspace(-4, 4, 100)
    grid2 = np.linspace(-4, 4, 100)
    pdf1 = stats.norm.pdf(grid1, loc=0, scale=1)
    pdf2 = stats.norm.pdf(grid2, loc=0, scale=1)

    jd = joint_density(grid1, pdf1, grid2, pdf2, rho=0.5)
    # Integrate: sum over 2D grid
    dx1 = grid1[1] - grid1[0]
    dx2 = grid2[1] - grid2[0]
    integral = jd.sum() * dx1 * dx2
    assert abs(integral - 1.0) < 0.1, f"Joint integral {integral} should be ~1"


def test_kendall_tau_formula():
    """Kendall's tau = (2/pi)*arcsin(rho) for Gaussian copula."""
    for rho in [-0.9, -0.5, 0.0, 0.3, 0.7, 0.99]:
        expected = 2.0 / np.pi * np.arcsin(rho)
        got = kendall_tau(rho)
        assert abs(got - expected) < 1e-10, f"rho={rho}: expected {expected}, got {got}"


def test_kendall_tau_zero_at_independence():
    """When rho=0, Kendall's tau should be 0."""
    assert abs(kendall_tau(0.0)) < 1e-15


def test_conditional_pdf_integrates_to_one():
    """Conditional PDF f(x2|x1) should integrate to ~1."""
    grid1 = np.linspace(-4, 4, 100)
    grid2 = np.linspace(-4, 4, 100)
    pdf1 = stats.norm.pdf(grid1, loc=0, scale=1)
    pdf2 = stats.norm.pdf(grid2, loc=0, scale=1)

    cond = conditional_pdf(grid1, pdf1, grid2, pdf2, rho=0.6)
    integral = np.trapezoid(cond, grid2)
    assert abs(integral - 1.0) < 0.1, f"Conditional integral {integral} should be ~1"


def test_full_copula_returns_all_keys():
    """gaussian_copula() should return all expected keys with correct shapes."""
    grid1 = np.linspace(-3, 3, 50)
    grid2 = np.linspace(0, 2, 40)
    pdf1 = stats.norm.pdf(grid1, loc=0, scale=1)
    pdf2 = stats.norm.pdf(grid2, loc=1, scale=0.3)

    result = gaussian_copula(grid1, pdf1, grid2, pdf2, rho=0.4)

    expected_keys = {
        "joint_density", "conditional_pdf_given_x1", "kendall_tau",
        "marginal1_cdf", "marginal2_cdf", "rho",
    }
    assert set(result.keys()) == expected_keys
    assert len(result["joint_density"]) == 50  # n1
    assert len(result["joint_density"][0]) == 40  # n2
    assert len(result["conditional_pdf_given_x1"]) == 40  # on grid2
    assert len(result["marginal1_cdf"]) == 50
    assert len(result["marginal2_cdf"]) == 40
    assert abs(result["rho"] - 0.4) < 1e-12
