"""Hand scores and failures, with no model service or human-label fabrication."""

import json

import pytest

from ai_quality_evals_advanced.evaluation import (
    Annotation,
    agreement,
    cases,
    controls,
    export_annotations,
    import_annotations,
    wilson,
)


def test_split_and_mechanical_scores():
    assert len(cases()) == 20 and len(cases("held_out")) == 10
    report = controls("all")
    assert len(report["cases"]) == 30
    assert (
        report["counts"]["normalized"]["passes"] >= report["counts"]["exact"]["passes"]
    )
    assert report["counts"]["normalized"]["false_pass_ids"]
    assert report["counts"]["exact"]["false_failure_ids"]


def test_annotations_require_real_entries(tmp_path):
    path = tmp_path / "review.json"
    export_annotations(path)
    assert import_annotations(path) == []
    data = json.loads(path.read_text())
    assert all(
        "label" not in item and "reference" not in item for item in data["items"]
    )
    a = Annotation(
        case_id="ds01",
        reviewer="test-fixture-reviewer",
        label="correct",
        reason="test fixture",
    )
    data["annotations"] = [a.model_dump(), a.model_dump()]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        import_annotations(path)


def test_hand_agreement_and_interval():
    rows = [
        Annotation(
            case_id=str(i), reviewer=r, label=label, reason="controlled unit-test label"
        )
        for r, labels in (
            ("a", ["correct", "incorrect"]),
            ("b", ["correct", "correct"]),
        )
        for i, label in enumerate(labels)
    ]
    result = agreement(rows, "a", "b")
    assert result["agreement"] == 0.5 and result["kappa"] == 0
    low, high = wilson(1, 2)
    assert low == pytest.approx(0.0945312) and high == pytest.approx(0.9054688)
    assert wilson(0, 0) is None
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_deepeval_executes_with_explicit_control_judge():
    from types import SimpleNamespace

    from ai_quality_evals_advanced.contracts import Reply
    from ai_quality_evals_advanced.judges import faithfulness

    class ControlInference:
        config = SimpleNamespace(model="authored-unit-test-judge")

        def generate(self, messages, schema):
            fields = schema["properties"]
            if "truths" in fields:
                value = {"truths": ["Backpack costs 800 INR"]}
            elif "claims" in fields:
                value = {"claims": ["Backpack costs 900 INR"]}
            elif "verdicts" in fields:
                value = {"verdicts": [{"verdict": "no", "reason": "Wrong price"}]}
            else:
                value = {"reason": "Wrong price"}
            return Reply(
                text=json.dumps(value), model=self.config.model, finish_reason="stop"
            )

    measured = faithfulness(ControlInference(), "Price?", "900 INR", ["800 INR"])
    assert measured["score"] == 0
