"""Credential-free tests for real-output capture, boundaries and replay."""

import io
import json
import sys
from pathlib import Path

import pytest

from ai_quality_evals import demoshop
from ai_quality_evals.foundations import Dataset
from ai_quality_evals.providers import ollama

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / demoshop.DEFAULT_FIXTURE
PROMPT = ROOT / demoshop.DEFAULT_PROMPT
QUESTIONS = ROOT / "intro/datasets/demoshop/answering-v1.json"


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    def deny_network(*args, **kwargs):
        raise AssertionError("Unit tests must not call a model service")

    monkeypatch.setattr(ollama, "urlopen", deny_network)
    monkeypatch.chdir(ROOT)


def sample_completion(text="The notebook costs INR 100."):
    return ollama.Completion(
        output=text,
        model="test-local:tiny",
        finish_reason="stop",
        prompt_tokens=50,
        completion_tokens=8,
        latency_ms=12.0,
        raw_response={"model": "test-local:tiny", "message": {"content": text}},
    )


def questions():
    return demoshop.QuestionSet.model_validate_json(QUESTIONS.read_bytes())


def run_sample(monkeypatch, cases=None):
    monkeypatch.setattr(
        ollama,
        "installed_model",
        lambda *args: {"name": "test-local:tiny", "digest": "test-digest"},
    )
    return demoshop.run_questions(
        FIXTURE,
        PROMPT,
        cases or questions(),
        model="test-local:tiny",
        base_url="http://localhost:11434",
        max_tokens=128,
        timeout=10,
    )


def test_prompt_contains_source_facts_without_goldens_or_private_orders():
    fixture = Dataset.model_validate_json(FIXTURE.read_bytes())
    messages = demoshop.build_messages(
        fixture, PROMPT.read_text(), "What is the price?"
    )
    system = messages[0]["content"]
    public_data = json.loads(system.split("SHOP DATA:\n", 1)[1])
    assert public_data["products"][0]["price_inr"] == 100
    assert set(public_data) == {"currency", "products", "policies"}
    assert "DS-1001" not in system and "C001" not in system
    assert "task_correct" not in system and "candidate" not in system
    assert messages[1] == {"role": "user", "content": "What is the price?"}


def test_http_request_and_answer_are_not_substituted_with_reference(monkeypatch):
    def respond(request, timeout):
        assert request.full_url == "http://localhost:11434/api/chat"
        payload = json.loads(request.data)
        assert payload["model"] == "test-local:tiny"
        assert payload["stream"] is False
        assert payload["options"]["num_predict"] == 64
        assert payload["options"]["temperature"] == 0
        return io.BytesIO(
            json.dumps(
                {
                    "model": "test-local:tiny",
                    "done": True,
                    "done_reason": "stop",
                    "message": {
                        "content": "A real response would be preserved verbatim."
                    },
                    "prompt_eval_count": 10,
                    "eval_count": 9,
                }
            ).encode()
        )

    monkeypatch.setattr(ollama, "urlopen", respond)
    result = ollama.chat("http://localhost:11434", "test-local:tiny", [], max_tokens=64)
    assert result.output == "A real response would be preserved verbatim."
    assert result.prompt_tokens == 10 and result.completion_tokens == 9


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "http://user:pass@localhost:11434",
        "http://localhost:11434/proxy",
        "http://localhost:11434?route=other",
    ],
)
def test_only_local_uncredentialed_endpoint_is_accepted(url):
    with pytest.raises(ValueError, match="local HTTP"):
        ollama.local_url(url, "/api/chat")


@pytest.mark.parametrize(
    "response",
    [
        {"done": False},
        {"done": True, "message": {"content": ""}},
        {"done": True, "message": {"content": " \n"}},
        {"done": True, "message": {"content": 42}},
    ],
)
def test_incomplete_or_empty_answers_are_run_errors(monkeypatch, response):
    monkeypatch.setattr(ollama, "request_json", lambda *args: response)
    with pytest.raises(RuntimeError):
        ollama.chat("http://localhost:11434", "test-local:tiny", [])


def test_missing_model_does_not_trigger_a_download(monkeypatch):
    def tags_only(url, payload, timeout):
        assert url.endswith("/api/tags") and payload is None
        return {"models": []}

    monkeypatch.setattr(ollama, "request_json", tags_only)
    with pytest.raises(ValueError, match="no download"):
        ollama.installed_model("http://localhost:11434", "missing-model")


def test_model_metadata_includes_digest_and_runtime_version(monkeypatch):
    def metadata(url, payload, timeout):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "test:tiny", "digest": "frozen-weights"}]}
        if url.endswith("/api/show"):
            assert payload == {"model": "test:tiny"}
            return {"capabilities": ["completion", "tools"]}
        assert url.endswith("/api/version")
        return {"version": "test-runtime"}

    monkeypatch.setattr(ollama, "request_json", metadata)
    result = ollama.installed_model("http://localhost:11434", "test:tiny")
    assert result["digest"] == "frozen-weights"
    assert result["ollama_version"] == "test-runtime"
    assert result["capabilities"] == ["completion", "tools"]


def test_native_tool_http_payload_and_empty_text_response(monkeypatch):
    tool = {"type": "function", "function": {"name": "get_product"}}
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_product",
                    "arguments": {"product_id": "SKU-NOTEBOOK"},
                }
            }
        ],
    }

    def respond(request, timeout):
        payload = json.loads(request.data)
        assert payload["tools"] == [tool] and "format" not in payload
        assert payload["think"] is False
        return io.BytesIO(
            json.dumps(
                {
                    "done": True,
                    "done_reason": "stop",
                    "model": "test:tools",
                    "message": message,
                }
            ).encode()
        )

    monkeypatch.setattr(ollama, "urlopen", respond)
    result = ollama.chat(
        "http://localhost:11434", "test:tools", [], tools=[tool], thinking=False
    )
    assert isinstance(result, ollama.ToolCompletion)
    assert result.output == "" and result.tool_calls == message["tool_calls"]
    assert result.raw_response["message"] == message


def test_reports_capture_actual_output_and_do_not_claim_semantic_accuracy(monkeypatch):
    monkeypatch.setattr(ollama, "chat", lambda *args, **kwargs: sample_completion())
    report = run_sample(monkeypatch)
    assert report["summary"]["completed"] == 6
    first = report["results"][0]
    assert first["case"]["reference"] == "INR 100"
    assert first["completion"]["output"] == "The notebook costs INR 100."
    assert first["matching_scores"] == {"exact": 0, "normalized": 0}
    assert "pending" in first["manual_review"]
    assert "not automatically graded" in report["summary"]["task_correctness"]
    assert report["provenance"]["installed_model"]["digest"] == "test-digest"
    assert report["provenance"]["fixture_sha256"]
    assert report["provenance"]["prompt_sha256"]


def test_partial_model_failure_is_retained_in_denominator(monkeypatch):
    responses = iter([sample_completion(), RuntimeError("Ollama timed out")])

    def complete(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(ollama, "chat", complete)
    subset = questions().model_copy(update={"cases": questions().cases[:2]})
    report = run_sample(monkeypatch, subset)
    assert report["summary"]["requested"] == 2
    assert report["summary"]["completed"] == 1
    assert report["summary"]["errors"] == 1
    assert report["results"][1]["completion"] is None
    assert report["results"][1]["matching_scores"] is None
    assert report["results"][1]["error"] == "Ollama timed out"


def test_recorded_replay_needs_no_model_or_network(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ollama, "chat", lambda *args, **kwargs: sample_completion())
    report = run_sample(monkeypatch)
    path = tmp_path / "recorded.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Replay must not invoke a model")

    monkeypatch.setattr(ollama, "chat", fail_if_called)
    monkeypatch.setattr(ollama, "installed_model", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["demoshop", "--replay", str(path)])
    assert demoshop.main() == 0
    assert "RECORDED OUTPUT REPLAY" in capsys.readouterr().out


def test_live_cli_requires_explicit_model_and_output(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demoshop", "--question", "Notebook price?"])
    with pytest.raises(SystemExit) as result:
        demoshop.main()
    assert result.value.code == 2
