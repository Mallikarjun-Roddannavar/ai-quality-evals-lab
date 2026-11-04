"""Rule goldens, metamorphic checks and offline recorded-answer evaluation."""

import json
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_quality_evals.catalog_evaluation import RuleSet, evaluate_recordings
from ai_quality_evals.evaluators.catalog import Product, check_catalog
from ai_quality_evals.foundations import sha256

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "intro/datasets/demoshop/foundations-v1.json"
RULES = ROOT / "intro/configs/evaluations/catalog-v1.json"
DATA = json.loads(FIXTURE.read_text(encoding="utf-8"))
PRODUCTS = {
    p["product_id"]: Product.model_validate(p)
    for p in DATA["shop_snapshot"]["products"]
}
PROBES = json.loads(
    (ROOT / "intro/datasets/demoshop/catalog-probes-v1.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.chdir(ROOT)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OLLAMA_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def deny(*args, **kwargs):
        raise AssertionError("Catalog evaluation must not access a model or network")

    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket.socket, "connect", deny)


@pytest.mark.parametrize("case", PROBES["cases"], ids=lambda c: c["case_id"])
def test_manual_probe_verdicts(case):
    result = check_catalog(case["answer"], PRODUCTS[case["product_id"]], case["check"])
    assert result.status == case["expected_status"]


@pytest.mark.parametrize(
    "answer",
    [
        "INR 800",
        "  inr\t800  ",
        "The BACKPACK costs 800 INR.",
        "800.00 rupees",
    ],
)
def test_price_preserving_format_changes(answer):
    assert check_catalog(answer, PRODUCTS["SKU-BACKPACK"], "price").status == "pass"


def test_change_in_price_changes_verdict_without_changing_format():
    product = PRODUCTS["SKU-BACKPACK"]
    assert check_catalog("INR 800", product, "price").status == "pass"
    assert check_catalog("INR 801", product, "price").status == "fail"


@pytest.mark.parametrize(
    "answer,check", [("INR 800?", "price"), ("in stock?", "stock")]
)
def test_questions_are_not_accepted_as_factual_assertions(answer, check):
    assert check_catalog(answer, PRODUCTS["SKU-BACKPACK"], check).status == "review"


def test_zero_to_positive_stock_changes_availability_expectation():
    zero = PRODUCTS["SKU-BOTTLE"]
    positive = Product(
        product_id=zero.product_id, name=zero.name, price_inr=300, stock=1
    )
    assert check_catalog("out of stock", zero, "stock").status == "pass"
    assert check_catalog("out of stock", positive, "stock").status == "fail"


@pytest.mark.parametrize("stock", [-1, True, "0"])
def test_reject_invalid_catalog_quantity(stock):
    with pytest.raises(ValidationError):
        Product(product_id="SKU-X", name="Example", price_inr=100, stock=stock)


def capture(tmp_path, name="recorded.json", *, error=False, finish="stop"):
    # Authored test double, never presented as a real captured model result.
    data = {
        "application": "DemoShop",
        "mode": "live",
        "provenance": {
            "fixture_sha256": sha256(FIXTURE.read_bytes()),
            "requested_model": "test-double",
        },
        "results": [
            {
                "case": {
                    "case_id": "interactive-1",
                    "question": "How much is the backpack?",
                },
                "completion": None
                if error
                else {"output": "The backpack costs 800 INR.", "finish_reason": finish},
                "error": "connection failed" if error else None,
            },
            {
                "case": {
                    "case_id": "policy",
                    "question": "Do you ship internationally?",
                },
                "completion": {"output": "Not available.", "finish_reason": "stop"},
                "error": None,
            },
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_fact_comparison_accepts_correct_sentence_that_string_matching_rejects(
    tmp_path,
):
    path = capture(tmp_path)
    report = evaluate_recordings([path], FIXTURE, RULES)
    assert report["summary"]["total"] == 2
    assert report["summary"]["pass"] == 1
    assert report["summary"]["not_evaluated"] == 1
    result = report["results"][0]
    assert result["status"] == "pass" and result["observed"] == "800"
    assert result["string_matches"] == {
        "reference": "INR 800",
        "exact": 0,
        "normalized": 0,
    }
    assert report["results"][1]["reason"] == "no_catalog_rule_for_question"


@pytest.mark.parametrize(
    "error,finish,status",
    [
        (True, "stop", "error"),
        (False, "length", "review"),
    ],
)
def test_generation_errors_and_truncation_do_not_pass(tmp_path, error, finish, status):
    path = capture(tmp_path, error=error, finish=finish)
    report = evaluate_recordings([path], FIXTURE, RULES)
    assert report["results"][0]["status"] == status
    assert report["results"][0]["string_matches"] is None


def test_report_fingerprint_is_repeatable_and_preserves_source_capture(tmp_path):
    path = capture(tmp_path)
    original = path.read_bytes()
    first = evaluate_recordings([path], FIXTURE, RULES)
    second = evaluate_recordings([path], FIXTURE, RULES)
    assert first["run_id"] == second["run_id"]
    assert first["results"] == second["results"]
    assert path.read_bytes() == original


def test_interactive_case_ids_are_qualified_by_input_report(tmp_path):
    one = capture(tmp_path, "one.json")
    two = capture(tmp_path, "two.json")
    report = evaluate_recordings([one, two], FIXTURE, RULES)
    assert report["summary"]["pass"] == 2
    assert report["results"][0]["input_report"] != report["results"][2]["input_report"]
    with pytest.raises(ValueError, match="double-count"):
        evaluate_recordings([one, one], FIXTURE, RULES)


def test_fixture_mismatch_prevents_scoring_against_changed_facts(tmp_path):
    path = capture(tmp_path)
    data = json.loads(path.read_text())
    data["provenance"]["fixture_sha256"] = "different-source-facts"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs"):
        evaluate_recordings([path], FIXTURE, RULES)


def test_question_rules_are_explicit_unique_and_validated():
    data = json.loads(RULES.read_text())
    data["rules"].append(data["rules"][0])
    with pytest.raises(ValidationError, match="one rule"):
        RuleSet.model_validate(data)
