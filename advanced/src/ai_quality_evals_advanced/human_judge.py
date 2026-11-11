"""Review real generated answers before comparing with an independent judge."""

import json
import os
import uuid
from datetime import UTC, datetime
from hashlib import sha256

from ai_quality_evals.providers.openrouter import load_dotenv
from ai_quality_evals_advanced.applications import public_facts
from ai_quality_evals_advanced.contracts import ARTIFACTS, ROOT, Settings, run_folder
from ai_quality_evals_advanced.evaluation import Annotation
from ai_quality_evals_advanced.providers import Inference, ProviderError
from ai_quality_evals_advanced.runtime import atomic_json
from ai_quality_evals_advanced.security import redact
from ai_quality_evals_advanced.telemetry import Telemetry

DATASET = ROOT / "advanced/datasets/human-judge-v1.json"
CONTROLS = ROOT / "advanced/datasets/human-judge-controls-v1.json"
RUBRIC = (
    "Correct means the answer is supported by the supplied facts and answers "
    "the question without inventing facts. If information is missing, an "
    "answer that admits the information is missing is correct."
)


def _read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def _digest(data):
    return sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _settings(model):
    load_dotenv()
    selected = model or os.environ.get("OPENROUTER_MODEL", "").strip()
    if not selected:
        raise ValueError("Set OPENROUTER_MODEL in .env or provide --model")
    return Settings(
        provider="openrouter",
        model=selected,
        max_model_calls=40,
        max_tokens=1024,
        deadline_seconds=300.0,
        request_timeout=60.0,
    )


def generate(model=None, limit=5, *, artifacts=ARTIFACTS):
    settings = _settings(model)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Add OPENROUTER_API_KEY to .env or the environment")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if not 1 <= limit <= len(dataset["cases"]):
        raise ValueError("Limit must be between 1 and the number of dataset cases")
    run_id = uuid.uuid4().hex
    folder = run_folder(run_id, artifacts)
    folder.mkdir(parents=True, exist_ok=False)
    facts = public_facts()
    telemetry = Telemetry(
        folder, run_id, settings.phoenix_endpoint, capture_content=True
    )
    inference = Inference(settings, telemetry)
    rows = []
    try:
        with telemetry.span("human_judge.generate"):
            for case in dataset["cases"][:limit]:
                row = {
                    "case_id": case["id"],
                    "question": case["question"],
                    "answer_origin": "model_generated",
                }
                try:
                    with telemetry.span(
                        "human_judge.generate.case", **{"case.id": case["id"]}
                    ) as span:
                        span.set_attribute("input.value", redact(case["question"]))
                        reply = inference.generate(
                            [
                                {
                                    "role": "system",
                                    "content": (
                                        "Answer using only these DemoShop facts. "
                                        "If information is missing, say so. "
                                        + json.dumps(facts)
                                    ),
                                },
                                {"role": "user", "content": case["question"]},
                            ]
                        )
                        span.set_attribute("output.value", redact(reply.text))
                        span.set_attribute("model.actual", reply.model)
                    if reply.finish_reason != "stop":
                        raise ValueError("incomplete_model_answer")
                    row.update(answer=reply.text, model=reply.model, status="completed")
                except Exception as exc:
                    error = (
                        str(exc)
                        if isinstance(exc, ProviderError)
                        else type(exc).__name__
                    )
                    row.update(status="execution_error", error=error)
                rows.append(row)
    finally:
        trace = telemetry.close()
    control_dataset = json.loads(CONTROLS.read_text(encoding="utf-8"))
    for control in control_dataset["controls"]:
        rows.append(
            {**control, "answer_origin": "authored_control", "status": "completed"}
        )
    bundle = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": dataset["version"],
        "dataset_sha256": sha256(DATASET.read_bytes()).hexdigest(),
        "controls_version": control_dataset["version"],
        "controls_sha256": sha256(CONTROLS.read_bytes()).hexdigest(),
        "rubric": RUBRIC,
        "facts": facts,
        "provider": "openrouter",
        "requested_model": settings.model,
        "cases": rows,
        "telemetry": trace,
    }
    atomic_json(folder / "responses.json", bundle)
    return bundle


def save_reviews(run_id, reviewer, assessments, *, artifacts=ARTIFACTS):
    folder = run_folder(run_id, artifacts)
    responses = _read(folder, "responses.json")
    expected = {
        row["case_id"] for row in responses["cases"] if row["status"] == "completed"
    }
    if not expected:
        raise ValueError("No completed answers to review")
    rows = []
    for item in assessments:
        rows.append(
            Annotation.model_validate({**item, "reviewer": reviewer}).model_dump()
        )
    actual = [row["case_id"] for row in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Review every completed case exactly once")
    path = folder / "human-review.json"
    if path.exists():
        raise ValueError("A review already exists for this run")
    atomic_json(path, {"response_sha256": _digest(responses), "annotations": rows})
    return rows


def judge(run_id, model=None, *, artifacts=ARTIFACTS):
    from ai_quality_evals_advanced.judges import python_judge

    folder = run_folder(run_id, artifacts)
    responses = _read(folder, "responses.json")
    review = _read(folder, "human-review.json")
    if review["response_sha256"] != _digest(responses):
        raise ValueError("Saved answers changed after human review")
    load_dotenv()
    settings = _settings(model or os.environ.get("OPENROUTER_JUDGE_MODEL"))
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("Add OPENROUTER_API_KEY to .env or the environment")
    telemetry = Telemetry(
        folder, run_id, settings.phoenix_endpoint, capture_content=True
    )
    inference = Inference(settings, telemetry)
    rows = []
    try:
        with telemetry.span("human_judge.judge"):
            for case in responses["cases"]:
                if case["status"] != "completed":
                    continue
                row = {"case_id": case["case_id"]}
                try:
                    with telemetry.span(
                        "human_judge.judge.case",
                        "EVALUATOR",
                        **{
                            "case.id": case["case_id"],
                            "answer.origin": case.get(
                                "answer_origin", "model_generated"
                            ),
                        },
                    ) as span:
                        span.set_attribute("input.value", redact(case["question"]))
                        span.set_attribute("case.answer", redact(case["answer"]))
                        result = python_judge(
                            inference,
                            case["question"],
                            case["answer"],
                            [json.dumps(responses["facts"]), RUBRIC],
                        )
                        span.set_attribute("output.value", json.dumps(redact(result)))
                    row.update(
                        result=result,
                        model=inference.metadata.get(
                            "last_response_model", settings.model
                        ),
                        label="correct"
                        if result["supported"] and result["complete"]
                        else "incorrect",
                        status="completed",
                    )
                except Exception as exc:
                    error = (
                        str(exc)
                        if isinstance(exc, ProviderError)
                        else type(exc).__name__
                    )
                    row.update(status="execution_error", error=error)
                rows.append(row)
    finally:
        trace = telemetry.close()
    output = {
        "response_sha256": _digest(responses),
        "judge_model": settings.model,
        "cases": rows,
        "telemetry": trace,
    }
    atomic_json(folder / f"judge-{uuid.uuid4().hex}.json", output)
    return output


def trace_saved(run_id, *, artifacts=ARTIFACTS):
    """Export a clearly marked replay of saved answers without model calls."""
    folder = run_folder(run_id, artifacts)
    responses = _read(folder, "responses.json")
    telemetry = Telemetry(
        folder, run_id, Settings().phoenix_endpoint, capture_content=True
    )
    try:
        with telemetry.span("human_judge.saved_answers", **{"trace.replay": True}):
            for case in responses["cases"]:
                with telemetry.span(
                    "human_judge.saved_answer",
                    **{"case.id": case["case_id"], "trace.replay": True},
                ) as span:
                    span.set_attribute("input.value", redact(case["question"]))
                    if case["status"] == "completed":
                        span.set_attribute("output.value", redact(case["answer"]))
                        if case.get("model"):
                            span.set_attribute("model.actual", case["model"])
                        span.set_attribute(
                            "answer.origin",
                            case.get("answer_origin", "model_generated"),
                        )
                    else:
                        span.set_attribute("execution.status", "execution_error")
    finally:
        summary = telemetry.close()
    return summary


def compare(run_id, *, artifacts=ARTIFACTS):
    folder = run_folder(run_id, artifacts)
    responses = _read(folder, "responses.json")
    review = _read(folder, "human-review.json")
    judge_paths = sorted(
        folder.glob("judge-*.json"), key=lambda path: path.stat().st_mtime_ns
    )
    if not judge_paths:
        raise ValueError("Run the judge stage before comparing")
    judge_results = json.loads(judge_paths[-1].read_text(encoding="utf-8"))
    if (
        _digest(responses) != review["response_sha256"]
        or _digest(responses) != judge_results["response_sha256"]
    ):
        raise ValueError("Saved answers do not match the reviewed and judged answers")
    by_case = {r["case_id"]: r for r in review["annotations"]}
    judged = {r["case_id"]: r for r in judge_results["cases"]}
    rows = []
    for case in responses["cases"]:
        human = by_case.get(case["case_id"])
        machine = judged.get(case["case_id"])
        human_label = human["label"] if human else None
        judge_label = machine.get("label") if machine else None
        comparable = human_label in {"correct", "incorrect"} and judge_label in {
            "correct",
            "incorrect",
        }
        rows.append(
            {
                "case_id": case["case_id"],
                "answer_origin": case.get("answer_origin", "model_generated"),
                "answer": case.get("answer"),
                "answer_model": case.get("model"),
                "human_label": human_label,
                "human_reason": human["reason"] if human else None,
                "judge_label": judge_label,
                "judge_model": machine.get("model") if machine else None,
                "judge_reason": machine.get("result", {}).get("reason")
                if machine
                else None,
                "agreement": human_label == judge_label if comparable else None,
                "generation_status": case["status"],
                "judge_status": machine["status"] if machine else "not_judged",
                "control_detected": (
                    judge_label == case["expected_judge_label"]
                    if case.get("answer_origin") == "authored_control"
                    and judge_label in {"correct", "incorrect"}
                    else None
                ),
            }
        )
    comparable = [
        r
        for r in rows
        if r["answer_origin"] == "model_generated" and r["agreement"] is not None
    ]
    controls = [r for r in rows if r["answer_origin"] == "authored_control"]
    controls_judged = [r for r in controls if r["control_detected"] is not None]
    return {
        "cases": rows,
        "compared": len(comparable),
        "total": len(rows) - len(controls),
        "controls_total": len(controls),
        "controls_judged": len(controls_judged),
        "controls_detected": sum(r["control_detected"] for r in controls_judged),
        "agreement": sum(r["agreement"] for r in comparable) / len(comparable)
        if comparable
        else None,
        "same_model_selection": responses["requested_model"]
        == judge_results["judge_model"],
        "limitation": (
            "One reviewer and a small set do not establish judge reliability; "
            "uncertain and errors are excluded from agreement."
        ),
    }
