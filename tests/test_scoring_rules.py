"""Tests for proper scoring rules module."""

import numpy as np
import pytest

from priorlab.scoring_rules import (
    log_score,
    crps,
    brier_score,
    skill_score,
    score_prior,
    comparative_scoring,
)
from priorlab.fitting import fit_all_distributions, select_best_fit
from priorlab.models import ExpertPrior, ElicitedQuantiles


class TestScoringRules:
    """Proper scoring rule tests."""

    def test_log_score_is_negative(self, symmetric_quantiles):
        """Log score should be negative for proper densities (pdf < 1 at most points)."""
        fits = fit_all_distributions(symmetric_quantiles)
        best = select_best_fit(fits)
        ls = log_score(symmetric_quantiles, best)
        assert ls < 0, f"Log score = {ls}, expected negative"

    def test_skill_score_bounded(self, symmetric_quantiles):
        """Skill score must be <= 1 (perfect), and typically > 0 for a good prior."""
        fits = fit_all_distributions(symmetric_quantiles)
        best = select_best_fit(fits)
        ss = skill_score(symmetric_quantiles, best)
        assert ss <= 1.0 + 1e-10, f"Skill score = {ss}, must be <= 1"
        # A well-fitted prior should beat Uniform
        assert ss > 0.0, f"Skill score = {ss}, expected > 0 for well-fitted prior"

    def test_crps_nonnegative(self, skewed_quantiles):
        """CRPS must be non-negative."""
        fits = fit_all_distributions(skewed_quantiles)
        best = select_best_fit(fits)
        cr = crps(skewed_quantiles, best)
        assert cr >= 0.0, f"CRPS = {cr}, expected >= 0"

    def test_brier_score_nonnegative(self, bounded_quantiles):
        """Brier score must be non-negative."""
        fits = fit_all_distributions(bounded_quantiles)
        best = select_best_fit(fits)
        bs = brier_score(bounded_quantiles, best)
        assert bs >= 0.0, f"Brier score = {bs}, expected >= 0"

    def test_comparative_ranking(self):
        """Comparative scoring should rank experts by total score."""
        eq1 = ElicitedQuantiles(lower=-1.0, q1=-0.5, median=0.0, q3=0.5, upper=1.0)
        eq2 = ElicitedQuantiles(lower=-2.0, q1=-1.0, median=0.0, q3=1.0, upper=2.0)
        fits1 = fit_all_distributions(eq1)
        fits2 = fit_all_distributions(eq2)
        expert1 = ExpertPrior(
            expert_id="E1", label="Expert 1",
            quantiles=eq1, best_fit=select_best_fit(fits1),
        )
        expert2 = ExpertPrior(
            expert_id="E2", label="Expert 2",
            quantiles=eq2, best_fit=select_best_fit(fits2),
        )
        rankings = comparative_scoring([expert1, expert2])
        assert len(rankings) == 2
        assert rankings[0]["total_score"] >= rankings[1]["total_score"]
        assert rankings[0]["expert_id"] in ("E1", "E2")
