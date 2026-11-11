"""Batch harness failures and declared splits, independent of model quality."""

import pytest

from ai_quality_evals_advanced import batch
from ai_quality_evals_advanced.contracts import Settings
from ai_quality_evals_advanced.telemetry import Telemetry


def test_versioned_application_split():
    assert len(batch.scenarios("development")) == 20
    assert len(batch.scenarios("held_out")) == 10
    assert {c.mode for c in batch.scenarios("all")} == {
        "chat",
        "rag",
        "single",
        "multi",
    }
    with pytest.raises(ValueError):
        batch.scenarios("unknown")


def test_stock_harness_calls_actual_checker(tmp_path, monkeypatch):
    case = next(
        c for c in batch.scenarios("all") if c.expectation["kind"] == "availability"
    )
    monkeypatch.setattr(batch, "scenarios", lambda split: [case])
    monkeypatch.setattr(batch, "chat", lambda *args: {"outputs": ["in stock"]})
    telemetry = Telemetry(tmp_path, "control", None)
    try:
        result = batch.run_batch(Settings(export_traces=False), telemetry, tmp_path)
        assert result["cases"][0]["verdict"] == "pass"
        assert result["counts"]["execution_error"] == 0
    finally:
        telemetry.close()
