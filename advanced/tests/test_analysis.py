"""Hand calculations, paired clustering, failed attempts and real CLI gates."""

import json

import pytest

from ai_quality_evals_advanced import analysis, cli
from ai_quality_evals_advanced.contracts import Settings
from ai_quality_evals_advanced.experiments import run_experiment


@pytest.mark.parametrize(
    "name,case",
    [
        ("application", {"execution_status": "execution_error"}),
        ("agents", {"error": "ProviderError"}),
        ("operations", {"execution_error": "ProviderError"}),
    ],
)
def test_incomplete_live_cases_do_not_report_completed(
    monkeypatch, tmp_path, name, case
):
    if name == "application":
        monkeypatch.setattr(
            "ai_quality_evals_advanced.batch.run_batch",
            lambda *args: {"cases": [case]},
        )
    elif name == "agents":
        monkeypatch.setattr(
            "ai_quality_evals_advanced.scenarios.application_comparison",
            lambda *args: {"cases": [case]},
        )
    else:
        monkeypatch.setattr(
            analysis, "operation_trials", lambda *args: {"cases": [case]}
        )
    result = run_experiment(
        name, "live", Settings(export_traces=False), artifacts=tmp_path
    )
    assert result["execution_status"] == "execution_error"


def test_incomplete_judge_does_not_report_completed(monkeypatch, tmp_path):
    from ai_quality_evals_advanced import judges

    monkeypatch.setattr(
        judges,
        "python_judge",
        lambda *args: {"supported": True, "complete": True, "reason": "control"},
    )

    def fail(*args):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(judges, "faithfulness", fail)
    result = run_experiment(
        "judges", "live", Settings(export_traces=False), artifacts=tmp_path
    )
    assert result["execution_status"] == "execution_error"
    assert result["result"]["answer_origin"] == "authored_control"
    assert result["result"]["same_subject_model"] is None


def test_foundations_trace_has_case_results_without_raw_content(tmp_path):
    result = run_experiment(
        "foundations", "offline", Settings(export_traces=False), artifacts=tmp_path
    )
    path = tmp_path / result["run_id"] / "traces.jsonl"
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        spans.append(json.loads(line))
    cases = [span for span in spans if span["name"].startswith("evaluation.case.")]
    assert len(cases) == 20
    first = cases[0]["attributes"]
    assert first["case.id"] == "ds01"
    assert first["check.exact"] == 1
    assert first["check.normalized"] == 1
    assert first["evaluation.snapshot"] is True
    assert "request" not in first and "answer" not in first
    assert len([s for s in spans if s["name"] == "evaluation.summary"]) == 2


def test_clustered_bootstrap_is_paired_and_reproducible():
    pairs = [
        {"case_id": "a", "left": 0, "right": 1},
        {"case_id": "a", "left": 0, "right": 1},
        {"case_id": "b", "left": 1, "right": 0},
    ]
    result = analysis.paired_bootstrap(pairs)
    assert result["clusters"] == 2 and result["trials"] == 3
    assert result["delta"] == 0
    assert result == analysis.paired_bootstrap(pairs)
    assert result["interval"] == [-1, 1]


def test_latency_includes_errors_and_unknown_usage():
    summary = analysis.operation_summary(
        [
            {"elapsed_ms": 10, "status": "error"},
            {
                "elapsed_ms": 30,
                "status": "completed",
                "prompt_tokens": 5,
                "completion_tokens": 2,
            },
        ]
    )
    assert summary["latency_p50_ms"] == 10 and summary["latency_p95_ms"] == 30
    assert summary["attempt_elapsed_ms"] == 40 and summary["errors"] == 1
    assert summary["token_coverage"] == 0.5 and summary["cost"] is None


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("baseline", True),
        ("wrong-price", False),
        ("wrong-policy", False),
        ("wrong-retrieval", False),
        ("wrong-authorization", False),
    ],
)
def test_controlled_gate_exit(monkeypatch, variant, expected):
    assert analysis.regression(variant)["gate_pass"] is expected
    monkeypatch.setattr("sys.argv", ["evals", "regression", "--variant", variant])
    assert cli.main() == (0 if expected else 1)


def test_evidence_replay_comparison_and_modes(tmp_path, monkeypatch):
    config = Settings(export_traces=False)
    a = run_experiment("statistics", "offline", config, artifacts=tmp_path)
    b = run_experiment("statistics", "offline", config, artifacts=tmp_path)
    assert a["execution_status"] == b["execution_status"] == "completed"
    assert "sources" not in a["provenance"]
    assert "source_changed_during_run" not in a["provenance"]
    assert a["provenance"]["fixture_sha256"]
    # Replay performs no writes or inference.
    path = tmp_path / b["run_id"] / "experiment.json"
    previous = path.read_bytes()
    assert analysis.read_evidence(b["run_id"], tmp_path)["run_id"] == b["run_id"]
    assert path.read_bytes() == previous
    real = analysis.read_evidence
    monkeypatch.setattr(analysis, "read_evidence", lambda id: real(id, tmp_path))
    assert analysis.compare_saved(a["run_id"], b["run_id"])["matched_task"]
    data = json.loads(previous)
    data["split"] = "held_out"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        analysis.compare_saved(a["run_id"], b["run_id"])
    wrong = run_experiment("judges", "offline", config, artifacts=tmp_path)
    assert wrong["execution_status"] == "execution_error"
