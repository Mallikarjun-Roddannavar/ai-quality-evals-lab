"""Hand-labeled M01 goldens, failure probes and offline reproduction checks."""

import json
import os
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_quality_evals.evaluators.matching import exact_match, normalized_match
from ai_quality_evals.foundations import (
    Dataset,
    Report,
    build_report,
    evaluate,
    git_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "intro/datasets/demoshop/foundations-v1.json"
RAW = json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_live_inference(monkeypatch):
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OCI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    def deny_network(*args, **kwargs):
        raise AssertionError("M01 must not open network connections")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)


@pytest.fixture
def dataset():
    return Dataset.model_validate(deepcopy(RAW))


@pytest.mark.parametrize("case", RAW["cases"], ids=lambda case: case["case_id"])
def test_manually_labeled_matching_goldens(case):
    assert (
        exact_match(case["candidate"], case["reference"]) == case["expected"]["exact"]
    )
    assert (
        normalized_match(case["candidate"], case["reference"])
        == (case["expected"]["normalized"])
    )


def test_hand_calculated_aggregate_and_oracle_errors(dataset):
    result = evaluate(dataset)
    assert result.total == 24
    assert result.disagreement_ids == [f"ds{i:02}" for i in range(9, 16)]
    assert result.normalized_only_passes == 7
    assert result.exact_only_passes == 0
    expected = {
        "exact": (8, 16, 8 / 24, 8, 0, 9, 7, 15 / 24),
        "normalized": (15, 9, 15 / 24, 13, 2, 4, 5, 18 / 24),
    }
    for rule, (passed, failed, rate, tp, fp, fn, tn, accuracy) in expected.items():
        metrics = result.metrics[rule]
        assert (metrics.passed, metrics.failed) == (passed, failed)
        assert metrics.pass_rate == pytest.approx(rate)
        assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (tp, fp, fn, tn)
        assert metrics.oracle_accuracy == pytest.approx(accuracy)
        assert metrics.golden_agreement_count == 24


def test_deliberate_failures_are_diagnosed(dataset):
    results = {r.case.case_id: r for r in evaluate(dataset).results}
    assert results["ds14"].oracle_diagnostics == {"exact": "tn", "normalized": "fp"}
    assert results["ds15"].oracle_diagnostics["normalized"] == "fp"
    assert results["ds16"].oracle_diagnostics == {"exact": "fn", "normalized": "fn"}
    assert results["ds23"].oracle_diagnostics == {"exact": "tn", "normalized": "tn"}
    assert results["ds24"].reasons["exact"] == "exact_mismatch"


def test_demoshop_snapshot_and_manual_references(dataset):
    snapshot = dataset.shop_snapshot
    products = {p["product_id"]: p for p in snapshot["products"]}
    orders = {o["order_id"]: o for o in snapshot["orders"]}
    cases = {c.case_id: c for c in dataset.cases}
    assert dataset.application == "DemoShop"
    assert dataset.application_stage == "authored_fixtures"
    assert dataset.fixture_version == "1.0.0"
    assert products["SKU-NOTEBOOK"]["price_inr"] == 100
    assert cases["ds01"].reference == "INR 100"
    assert products["SKU-BOTTLE"]["stock"] == 0
    assert cases["ds03"].reference == "out of stock"
    assert snapshot["policies"]["return_window_days"] == 14
    assert cases["ds05"].reference == "14 days"
    assert orders["DS-1002"]["status"] == cases["ds07"].reference == "shipped"
    assert cases["ds08"].reference == "not allowed"


@pytest.mark.parametrize("candidate", [None, 42, False, [], {}])
def test_reject_nonstring_outputs(candidate):
    data = deepcopy(RAW)
    data["cases"][0]["candidate"] = candidate
    with pytest.raises(ValidationError):
        Dataset.model_validate(data)


@pytest.mark.parametrize("score", [-1, 2, True, 1.0, "1"])
def test_scores_are_strict_binary_integers(score):
    data = deepcopy(RAW)
    data["cases"][0]["expected"]["exact"] = score
    with pytest.raises(ValidationError):
        Dataset.model_validate(data)


@pytest.mark.parametrize(
    "problem",
    ["empty", "duplicate", "extra", "blank", "missing", "app", "stage", "snapshot"],
)
def test_reject_invalid_dataset_contract(problem):
    data = deepcopy(RAW)
    if problem == "empty":
        data["cases"] = []
    elif problem == "duplicate":
        data["cases"].append(deepcopy(data["cases"][0]))
    elif problem == "extra":
        data["cases"][0]["candiate"] = "typo"
    elif problem == "blank":
        data["cases"][0]["case_id"] = " \t"
    elif problem == "app":
        data["application"] = "OtherApp"
    elif problem == "stage":
        data["application_stage"] = "live"
    elif problem == "snapshot":
        data["shop_snapshot"] = {}
    else:
        del data["cases"][0]["expected"]
    with pytest.raises(ValidationError):
        Dataset.model_validate(data)


def test_false_golden_is_detected_instead_of_used_as_score():
    data = deepcopy(RAW)
    data["cases"][0]["expected"]["exact"] = 0
    evaluation = evaluate(Dataset.model_validate(data))
    assert evaluation.results[0].scores.exact == 1
    assert evaluation.metrics["exact"].golden_agreement_count == 23


def test_baseline_and_framework_case_scores_agree_without_pydantic(dataset):
    # -S skips site-packages: the baseline can run without Pydantic installed.
    env = dict(os.environ, PYTHONPATH=str(ROOT / "intro/src"))
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "intro/tests/baselines/foundations_baseline.py",
            "--dataset",
            str(DATASET_PATH),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    baseline = json.loads(completed.stdout)
    framework = [
        {"case_id": r.case.case_id, **r.scores.model_dump()}
        for r in evaluate(dataset).results
    ]
    assert baseline == framework


def run_cli(dataset_path, output):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_quality_evals.foundations",
            "--dataset",
            str(dataset_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_repeated_cli_reports_are_reproducible_and_roundtrip(tmp_path):
    reports = []
    for index in range(2):
        output = tmp_path / f"run-{index}.json"
        completed = run_cli(DATASET_PATH, output)
        assert completed.returncode == 0, completed.stderr
        reports.append(Report.model_validate_json(output.read_bytes()))
    assert reports[0].run_id == reports[1].run_id
    assert reports[0].evaluation == reports[1].evaluation
    assert reports[0].provenance == reports[1].provenance
    assert reports[0].generated_at_utc.endswith("+00:00")
    sources = reports[0].provenance["source_sha256"]
    assert "uv.lock" in sources
    assert "intro/src/ai_quality_evals/evaluators/matching.py" in sources
    assert reports[0].provenance["application"] == "DemoShop"
    assert reports[0].provenance["shop_snapshot"] == RAW["shop_snapshot"]


def test_dataset_byte_change_is_visible_in_provenance(tmp_path):
    original = build_report(DATASET_PATH, ROOT)
    changed = tmp_path / "changed.json"
    changed.write_bytes(DATASET_PATH.read_bytes() + b"\n")
    report = build_report(changed, ROOT)
    assert original.evaluation == report.evaluation
    assert original.run_id != report.run_id
    assert original.provenance["dataset_sha256"] != report.provenance["dataset_sha256"]
    assert changed.as_posix() in report.provenance["reproduction_command"]


def test_missing_git_metadata_is_explicit_and_not_created(tmp_path):
    assert git_provenance(tmp_path) == {
        "revision": None,
        "dirty": None,
        "reason": "no_git_metadata",
    }
    assert not (tmp_path / ".git").exists()


def test_invalid_input_does_not_write_report(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"cases": []}', encoding="utf-8")
    output = tmp_path / "report.json"
    result = run_cli(invalid, output)
    assert result.returncode == 1
    assert "Evaluation failed:" in result.stderr
    assert not output.exists()


def test_cli_returns_failure_for_golden_mismatch(tmp_path):
    data = deepcopy(RAW)
    data["cases"][0]["expected"]["exact"] = 0
    path = tmp_path / "bad-golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "diagnostic.json"
    result = run_cli(path, output)
    assert result.returncode == 2
    report = Report.model_validate_json(output.read_bytes())
    assert report.evaluation.metrics["exact"].golden_agreement_count == 23
