import json
from types import SimpleNamespace

import pytest

from ai_quality_evals_advanced import providers
from ai_quality_evals_advanced.contracts import Settings
from ai_quality_evals_advanced.domain import ToolGateway
from ai_quality_evals_advanced.telemetry import Telemetry, record_evaluation


def test_openrouter_tool_history_and_model_logs(monkeypatch, tmp_path):
    captured = {}

    def fake_request(url, payload, timeout, key=None):
        captured.update(payload)
        return {
            "model": "example/model",
            "choices": [{"message": {"content": "INR 100"}, "finish_reason": "stop"}],
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "request", fake_request)
    telemetry = Telemetry(tmp_path, "test", None, capture_content=True)
    model = providers.Inference(Settings(provider="openrouter"), telemetry)
    history = [
        {"role": "user", "content": "price?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "get_product",
                        "arguments": {"product_id": "SKU-NOTEBOOK"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "100"},
    ]
    try:
        model.generate(history, tools=[])
    finally:
        telemetry.close()
    assert captured["messages"][1]["content"] is None
    assert json.loads(
        captured["messages"][1]["tool_calls"][0]["function"]["arguments"]
    ) == {"product_id": "SKU-NOTEBOOK"}
    assert captured["provider"]["require_parameters"] is True
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["messages"][0]["content"] == "price?"
    assert events[1]["response"]["text"] == "INR 100"
    assert "test-key" not in (tmp_path / "events.jsonl").read_text()


def test_provider_failure_is_recorded(monkeypatch, tmp_path):
    def fake_request(*args, **kwargs):
        raise providers.ProviderError("http_400")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "request", fake_request)
    telemetry = Telemetry(tmp_path, "test", None, capture_content=True)
    try:
        with pytest.raises(providers.ProviderError, match="http_400"):
            providers.Inference(Settings(provider="openrouter"), telemetry).generate(
                [{"role": "user", "content": "price?"}]
            )
    finally:
        telemetry.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["error"] == "http_400"


def test_tool_inputs_and_results_are_logged(tmp_path):
    telemetry = Telemetry(tmp_path, "test", None, capture_content=True)
    session = SimpleNamespace(get_product=lambda product_id: {"price_inr": 100})
    try:
        gateway = ToolGateway(session, {"get_product"}, telemetry=telemetry)
        gateway.call("get_product", {"product_id": "SKU-NOTEBOOK"})
    finally:
        telemetry.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["arguments"] == {"product_id": "SKU-NOTEBOOK"}
    assert events[1]["result"] == {"price_inr": 100}
    assert events[0]["span_id"] == events[1]["span_id"]


def test_benchmark_records_individual_case_output(tmp_path):
    telemetry = Telemetry(tmp_path, "test", None)
    try:
        record_evaluation(
            telemetry,
            "benchmark",
            {
                "models": [
                    {
                        "model": "example",
                        "inspect": [
                            {"samples": [{"id": "notebook", "output": "INR 100"}]}
                        ],
                    }
                ]
            },
        )
    finally:
        telemetry.close()
    spans = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl").read_text().splitlines()
    ]
    sample = next(s for s in spans if s["name"] == "evaluation.benchmark.case")
    assert json.loads(sample["attributes"]["output.value"])["output"] == "INR 100"
