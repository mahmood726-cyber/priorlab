import json
import pytest
from priorlab.pipeline import run_priorlab
from priorlab.models import PriorLabResult, ElicitedQuantiles


def test_pipeline_single_expert(symmetric_quantiles):
    result = run_priorlab(
        quantiles_list=[symmetric_quantiles],
        expert_labels=["Expert 1"],
        parameter_name="theta",
    )
    assert isinstance(result, PriorLabResult)
    assert len(result.experts) == 1
    assert result.experts[0].best_fit is not None


def test_pipeline_multi_expert():
    q1 = ElicitedQuantiles(lower=-0.8, q1=-0.4, median=-0.2, q3=0.0, upper=0.3)
    q2 = ElicitedQuantiles(lower=-0.6, q1=-0.3, median=-0.15, q3=0.05, upper=0.4)
    result = run_priorlab(quantiles_list=[q1, q2], expert_labels=["E1", "E2"])
    assert result.aggregated is not None
    assert len(result.aggregated.pdf_values) > 0


def test_pipeline_export_json_valid(symmetric_quantiles):
    result = run_priorlab(quantiles_list=[symmetric_quantiles], expert_labels=["E1"])
    export = result.export_json
    assert "family" in export
    assert "params" in export
    assert "parameter" in export


def test_pipeline_certification(symmetric_quantiles):
    result = run_priorlab(quantiles_list=[symmetric_quantiles], expert_labels=["E1"])
    assert result.certification in ("PASS", "WARN", "REJECT")


def test_pipeline_hash_nonempty(symmetric_quantiles):
    result = run_priorlab(quantiles_list=[symmetric_quantiles], expert_labels=["E1"])
    assert len(result.input_hash) > 0
