from types import SimpleNamespace

import pytest

from ai_quality_evals_advanced.contracts import Settings
from ai_quality_evals_advanced.scenarios import application_comparison


def test_benchmark_uses_requested_openrouter_models(monkeypatch, tmp_path):
    selected = []

    class FakeInference:
        def __init__(self, config, telemetry):
            self.config = config
            self.telemetry = telemetry
            self.attempts = []
            self.metadata = {"model": config.model}

    def fake_benchmark(subject, samples, folder):
        selected.append((subject.config.model, folder.name))
        return [{"status": "success", "samples": []}]

    monkeypatch.setattr("ai_quality_evals_advanced.providers.Inference", FakeInference)
    monkeypatch.setattr(
        "ai_quality_evals_advanced.frameworks.inspect_benchmark", fake_benchmark
    )
    inference = SimpleNamespace(config=Settings(provider="openrouter"), telemetry=None)
    result = application_comparison(
        "benchmark", inference, tmp_path, "test", "live", ["org/a:free", "org/b:free"]
    )
    assert selected == [("org/a:free", "model-1"), ("org/b:free", "model-2")]
    assert [row["model"] for row in result["models"]] == ["org/a:free", "org/b:free"]


def test_openrouter_benchmark_requires_models(tmp_path):
    inference = SimpleNamespace(config=Settings(provider="openrouter"), telemetry=None)
    with pytest.raises(ValueError, match="two OpenRouter model IDs"):
        application_comparison("benchmark", inference, tmp_path, "test", "live")
