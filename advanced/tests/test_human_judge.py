"""Exercise the connected workflow without an API key or live inference."""

import json
from types import SimpleNamespace

import pytest

from ai_quality_evals_advanced import human_judge, judges, providers
from ai_quality_evals_advanced.contracts import Reply, Settings


def test_live_answer_is_reviewed_and_judged_once(tmp_path, monkeypatch):
    class FakeInference:
        def __init__(self, config, telemetry):
            self.config = config
            self.metadata = {"last_response_model": "fake"}

        def generate(self, messages, schema=None):
            return Reply(
                text="The notebook costs 100 INR.", model="fake", finish_reason="stop"
            )

    seen = []

    def fake_judge(inference, question, answer, context):
        seen.append((question, answer, context))
        return {"supported": True, "complete": True, "reason": "Matches facts."}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(human_judge, "Inference", FakeInference)
    monkeypatch.setattr(judges, "python_judge", fake_judge)
    monkeypatch.setattr(
        human_judge,
        "_settings",
        lambda model: Settings(
            provider="openrouter", model="fake", export_traces=False
        ),
    )
    # Do not export telemetry during an offline test.
    from ai_quality_evals_advanced.telemetry import Telemetry

    monkeypatch.setattr(
        human_judge,
        "Telemetry",
        lambda folder, run_id, endpoint, **kwargs: Telemetry(
            folder, run_id, None, **kwargs
        ),
    )
    run = human_judge.generate("fake", 1, artifacts=tmp_path)
    run_id = run["run_id"]
    spans = [
        json.loads(line)
        for line in (tmp_path / run_id / "traces.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    case_span = next(
        span for span in spans if span["name"] == "human_judge.generate.case"
    )
    assert case_span["attributes"]["input.value"] == run["cases"][0]["question"]
    assert case_span["attributes"]["output.value"] == run["cases"][0]["answer"]
    assert run["telemetry"]["raw_content_capture"] is True
    with pytest.raises(FileNotFoundError):
        human_judge.judge(run_id, "fake", artifacts=tmp_path)
    human_judge.save_reviews(
        run_id,
        "Reviewer",
        [
            {
                "case_id": "price-notebook",
                "label": "correct",
                "reason": "Matches the catalog.",
            },
            {
                "case_id": "control-notebook-wrong-price",
                "label": "incorrect",
                "reason": "The catalog price is 100 INR.",
            },
        ],
        artifacts=tmp_path,
    )
    human_judge.judge(run_id, "fake", artifacts=tmp_path)
    result = human_judge.compare(run_id, artifacts=tmp_path)
    assert seen[0][1] == run["cases"][0]["answer"]
    assert result["compared"] == 1 and result["agreement"] == 1
    assert result["controls_total"] == 1 and result["controls_detected"] == 0
    assert result["cases"][1]["answer_origin"] == "authored_control"
    assert result["cases"][1]["control_detected"] is False
    assert seen[1][1] == "The notebook costs 900 INR."
    assert all("expected_judge_label" not in str(item) for item in seen[1])
    assert result["cases"][0]["human_label"] == result["cases"][0]["judge_label"]
    summary = human_judge.trace_saved(run_id, artifacts=tmp_path)
    assert summary["raw_content_capture"] is True
    spans = [
        json.loads(line)
        for line in (tmp_path / run_id / "traces.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    replay = next(span for span in spans if span["name"] == "human_judge.saved_answer")
    assert replay["attributes"]["trace.replay"] is True
    assert replay["attributes"]["output.value"] == run["cases"][0]["answer"]
    path = tmp_path / run_id / "responses.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cases"][0]["answer"] = "Changed answer"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        human_judge.compare(run_id, artifacts=tmp_path)


def test_openrouter_schema_is_sent(monkeypatch):
    captured = {}

    def fake_request(url, payload, timeout, key=None):
        captured.update(payload)
        return {
            "model": "fake",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        }

    class FakeSpan:
        def __enter__(self):
            return SimpleNamespace(
                set_attribute=lambda *args: None, set_status=lambda *args: None
            )

        def __exit__(self, *args):
            return False

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(providers, "request", fake_request)
    inference = providers.Inference(
        Settings(provider="openrouter", model="fake"),
        SimpleNamespace(span=lambda *args, **kwargs: FakeSpan()),
    )
    schema = {"type": "object", "properties": {}}
    inference.generate([{"role": "user", "content": "test"}], schema=schema)
    assert captured["response_format"]["json_schema"]["schema"] == schema
    assert captured["provider"]["require_parameters"] is True


def test_uncertain_and_generation_errors_do_not_count_as_agreement(tmp_path):
    folder = tmp_path / ("a" * 32)
    folder.mkdir()
    responses = {
        "requested_model": "fake",
        "cases": [
            {"case_id": "one", "answer": "Maybe", "status": "completed"},
            {"case_id": "two", "status": "execution_error"},
        ],
    }
    digest = human_judge._digest(responses)
    review = {
        "response_sha256": digest,
        "annotations": [{"case_id": "one", "label": "uncertain", "reason": "Not sure"}],
    }
    judge = {
        "response_sha256": digest,
        "judge_model": "fake",
        "cases": [
            {
                "case_id": "one",
                "status": "completed",
                "label": "correct",
                "result": {"reason": "Supported"},
            }
        ],
    }
    for name, value in (
        ("responses.json", responses),
        ("human-review.json", review),
        ("judge-test.json", judge),
    ):
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    result = human_judge.compare("a" * 32, artifacts=tmp_path)
    assert result["compared"] == 0 and result["agreement"] is None
    assert result["cases"][1]["judge_status"] == "not_judged"


def test_correct_rejection_of_control_is_counted_separately(tmp_path):
    folder = tmp_path / ("c" * 32)
    folder.mkdir()
    responses = {
        "requested_model": "fake",
        "cases": [
            {
                "case_id": "control",
                "answer": "900 INR",
                "answer_origin": "authored_control",
                "expected_judge_label": "incorrect",
                "status": "completed",
            }
        ],
    }
    digest = human_judge._digest(responses)
    review = {
        "response_sha256": digest,
        "annotations": [
            {"case_id": "control", "label": "incorrect", "reason": "Wrong price"}
        ],
    }
    judge = {
        "response_sha256": digest,
        "judge_model": "fake",
        "cases": [
            {
                "case_id": "control",
                "status": "completed",
                "label": "incorrect",
                "result": {"reason": "Wrong price"},
            }
        ],
    }
    for name, value in (
        ("responses.json", responses),
        ("human-review.json", review),
        ("judge-test.json", judge),
    ):
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    result = human_judge.compare("c" * 32, artifacts=tmp_path)
    assert result["controls_detected"] == result["controls_judged"] == 1
    assert result["compared"] == 0 and result["agreement"] is None


def test_cli_review_saves_human_label_before_judge(tmp_path, monkeypatch, capsys):
    import sys

    from ai_quality_evals_advanced import cli, contracts

    run_id = "b" * 32
    folder = tmp_path / run_id
    folder.mkdir()
    responses = {
        "run_id": run_id,
        "facts": {"products": []},
        "rubric": "Check source support.",
        "cases": [
            {
                "case_id": "x",
                "question": "Question?",
                "answer": "Answer.",
                "status": "completed",
            }
        ],
    }
    (folder / "responses.json").write_text(json.dumps(responses), encoding="utf-8")
    monkeypatch.setattr(contracts, "ARTIFACTS", tmp_path)
    human_judge_save = human_judge.save_reviews
    monkeypatch.setattr(
        human_judge,
        "save_reviews",
        lambda rid, reviewer, assessments: human_judge_save(
            rid, reviewer, assessments, artifacts=tmp_path
        ),
    )
    values = iter(["Reviewer", "correct", "Supported"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    monkeypatch.setattr(sys, "argv", ["evals", "human-judge", "review", run_id])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "Answer." in output
    assert not list(folder.glob("judge-*.json"))
    assert (
        json.loads((folder / "human-review.json").read_text())["annotations"][0][
            "label"
        ]
        == "correct"
    )


def test_cli_failed_judge_does_not_suggest_comparison(monkeypatch, capsys):
    import sys

    from ai_quality_evals_advanced import cli

    monkeypatch.setattr(
        human_judge,
        "judge",
        lambda *_args: {
            "cases": [
                {"case_id": "one", "status": "execution_error", "error": "empty_answer"}
            ]
        },
    )
    monkeypatch.setattr(sys, "argv", ["evals", "human-judge", "judge", "a" * 32])
    assert cli.main() == 1
    output = capsys.readouterr().out
    assert "Judge incomplete" in output
    assert "Next: evals human-judge compare" not in output
