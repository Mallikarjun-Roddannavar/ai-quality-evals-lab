"""Named experiments, immutable case evidence and explicit execution modes."""

import json
import time
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version

from ai_quality_evals_advanced.contracts import ARTIFACTS, FIXTURE, ROOT, Settings
from ai_quality_evals_advanced.evaluation import controls
from ai_quality_evals_advanced.providers import Inference
from ai_quality_evals_advanced.runtime import atomic_json
from ai_quality_evals_advanced.security import redact
from ai_quality_evals_advanced.telemetry import Telemetry, record_evaluation


def run_experiment(
    name: str,
    mode: str,
    config: Settings,
    split="development",
    artifacts=ARTIFACTS,
    benchmark_models=None,
) -> dict:
    if mode not in {"offline", "live"}:
        raise ValueError("Explicit offline/live mode required")
    identifier = uuid.uuid4().hex
    folder = artifacts / identifier
    folder.mkdir(parents=True, exist_ok=False)
    telemetry = Telemetry(
        folder,
        identifier,
        config.phoenix_endpoint if config.export_traces else None,
        capture_content=True,
    )
    inference = Inference(config, telemetry)
    started = time.perf_counter()
    pending_annotations = []
    data = {
        "schema_version": "2.0.0",
        "run_id": identifier,
        "experiment": name,
        "mode": mode,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution_status": "completed",
        "evidence_kind": "live"
        if mode == "live"
        else "local_mechanism"
        if name in {"mcp", "skills", "retrieval"}
        else "authored_control",
        "split": split,
    }
    try:
        with telemetry.span("experiment." + name) as root:
            data["trace_id"] = f"{root.get_span_context().trace_id:032x}"
            root.set_attribute("experiment.name", name)
            root.set_attribute("experiment.mode", mode)
            root.set_attribute("experiment.split", split)
            if name == "promptfoo":
                if mode != "offline":
                    raise ValueError("Tool security suite uses no model")
                from ai_quality_evals_advanced.frameworks import promptfoo_security

                data["result"] = promptfoo_security(folder)
            elif name == "application":
                if mode != "live":
                    raise ValueError("Application suite requires live mode")
                from ai_quality_evals_advanced.batch import run_batch

                data["result"] = run_batch(config, telemetry, folder, split)
            elif name == "mcp-selection":
                if mode != "live":
                    raise ValueError("MCP selection requires live mode")
                from ai_quality_evals_advanced.mcp_selection import run_selection

                data["result"] = run_selection(inference, folder, identifier)
            elif name in {"statistics", "regression", "operations", "bias"}:
                from ai_quality_evals_advanced.analysis import (
                    operation_trials,
                    regression,
                    statistics_report,
                )

                if name == "statistics":
                    if mode != "offline":
                        raise ValueError("Statistics uses authored controls")
                    data["result"] = statistics_report(split)
                elif name == "regression":
                    if mode != "offline":
                        raise ValueError("Regression uses controlled variants")
                    data["result"] = {
                        "variants": [
                            regression(v)
                            for v in (
                                "baseline",
                                "wrong-price",
                                "wrong-policy",
                                "wrong-retrieval",
                                "wrong-authorization",
                            )
                        ]
                    }
                elif name == "operations":
                    if mode != "live":
                        raise ValueError("Operations requires measured live requests")
                    data["result"] = operation_trials(inference)
                else:
                    if mode != "live":
                        raise ValueError("Bias probe requires live judgments")
                    from ai_quality_evals_advanced.judges import preference_probe

                    data["result"] = preference_probe(
                        inference,
                        "What is the backpack price?",
                        [
                            "800 INR",
                            "The premium backpack costs 900 INR, a great deal!",
                        ],
                        ["Backpack costs 800 INR."],
                    )
            elif name == "foundations":
                if mode != "offline":
                    raise ValueError(
                        "Foundations scores authored controls; choose offline"
                    )
                data["result"] = controls(split)
            elif name == "judges":
                if mode != "live":
                    raise ValueError(
                        "Judge experiment requires live mode; no simulated fallback"
                    )
                from ai_quality_evals_advanced.judges import faithfulness, python_judge

                facts = [
                    json.dumps(
                        json.loads(FIXTURE.read_bytes())["shop_snapshot"]["products"]
                    )
                ]
                results = []
                for answer in (
                    "The backpack costs 800 INR.",
                    "The backpack costs 900 INR.",
                ):
                    row = {"answer": answer}
                    for criterion, grader in (
                        ("python", python_judge),
                        ("deepeval", faithfulness),
                    ):
                        try:
                            with telemetry.span(
                                "judge." + criterion, "EVALUATOR"
                            ) as span:
                                row[criterion] = grader(
                                    inference,
                                    "What is the backpack price?",
                                    answer,
                                    facts,
                                )
                                grade = row[criterion]
                                passed = (
                                    grade.get("score", 0) >= 0.8
                                    if criterion == "deepeval"
                                    else (grade["supported"] and grade["complete"])
                                )
                                pending_annotations.append(
                                    (
                                        f"{span.get_span_context().span_id:016x}",
                                        "pass" if passed else "fail",
                                        grade["reason"],
                                        criterion,
                                    )
                                )
                        except Exception as exc:
                            row[criterion] = {"execution_error": type(exc).__name__}
                    results.append(row)
                data["result"] = {
                    "cases": results,
                    "judge_model": config.model,
                    "human_calibration": None,
                    "same_subject_model": None,
                    "answer_origin": "authored_control",
                }
                if any(
                    "execution_error" in row[criterion]
                    for row in results
                    for criterion in ("python", "deepeval")
                ):
                    data["execution_status"] = "execution_error"
                    data["error_detail"] = "One or more judge checks did not complete"
            elif name in {
                "chat",
                "rag",
                "benchmark",
                "tools",
                "mcp",
                "skills",
                "retrieval",
                "agents",
                "security",
            }:
                from ai_quality_evals_advanced.scenarios import application_comparison

                data["result"] = application_comparison(
                    name, inference, folder, identifier, mode, benchmark_models
                )
                if name == "benchmark" and not data["result"]["complete"]:
                    data["execution_status"] = "execution_error"
                    data["error_detail"] = (
                        "One or more benchmark models did not complete"
                    )
            else:
                raise ValueError("Unknown experiment")
            if name in {
                "agents",
                "security",
                "operations",
                "application",
                "mcp-selection",
            }:
                for case in data["result"].get("cases", []):
                    if (
                        case.get("execution_status") == "execution_error"
                        or case.get("execution_error")
                        or (name in {"agents", "security"} and case.get("error"))
                    ):
                        data["execution_status"] = "execution_error"
                        data["error_detail"] = "One or more cases did not complete"
                        break
            for group in ("cases", "variants", "models"):
                rows = data["result"].get(group)
                if isinstance(rows, list):
                    root.set_attribute("result." + group + "_count", len(rows))
            record_evaluation(telemetry, name, data["result"])
            root.set_attribute("execution.status", data["execution_status"])
    except Exception as exc:
        data.update(
            execution_status="execution_error",
            error=type(exc).__name__,
            error_detail=redact(str(exc)),
        )
    finally:
        data["telemetry"] = telemetry.close()
    if config.export_traces:
        data["annotations"] = [
            telemetry.annotate(
                config.phoenix_endpoint,
                span,
                verdict,
                reason,
                criterion="judge_" + criterion,
                kind="LLM",
            )
            for span, verdict, reason, criterion in pending_annotations
        ]
    data["elapsed_ms"] = (time.perf_counter() - started) * 1000
    data["attempts"] = inference.attempts
    from ai_quality_evals_advanced.analysis import operation_summary

    attempts = list(inference.attempts)
    result = data.get("result", {})
    for row in result.get("cases", []) + result.get("models", []):
        attempts.extend(row.get("attempts", []))
    data["usage"] = operation_summary(attempts)
    data["provider_metadata"] = inference.metadata
    data["provenance"] = {
        "fixture_sha256": sha256(FIXTURE.read_bytes()).hexdigest(),
        "lock_sha256": sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "npm_lock_sha256": sha256(
            (ROOT / "advanced/package-lock.json").read_bytes()
        ).hexdigest(),
        "packages": {
            p: version(p)
            for p in (
                "pydantic",
                "deepeval",
                "ragas",
                "inspect-ai",
                "langchain",
                "langgraph",
            )
        },
        "config": config.model_dump(),
        "full_module_validated": False,
    }
    atomic_json(folder / "experiment.json", redact(data))
    return data
