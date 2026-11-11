"""A1 price experiment: explicit controls/live execution and durable evidence."""

import json
import os
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from ai_quality_evals.evaluators.catalog import Product, check_catalog
from ai_quality_evals.foundations import git_provenance
from ai_quality_evals_advanced.contracts import (
    ARTIFACTS,
    FIXTURE,
    ROOT,
    PriceCase,
    RunResult,
    Settings,
    run_folder,
)
from ai_quality_evals_advanced.domain import ShopStore, ToolGateway
from ai_quality_evals_advanced.providers import Inference
from ai_quality_evals_advanced.security import redact
from ai_quality_evals_advanced.telemetry import Telemetry


def atomic_json(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(redact(data), stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_case() -> PriceCase:
    return PriceCase.model_validate_json(
        (ROOT / "advanced/datasets/price.json").read_bytes()
    )


def load_settings(path: Path | None = None) -> Settings:
    return Settings.model_validate_json(
        (path or ROOT / "advanced/configs/local.json").read_bytes()
    )


def execute(
    case: PriceCase, mode: str, config: Settings, *, artifacts: Path = ARTIFACTS
) -> RunResult:
    if mode not in {"offline", "live"}:
        raise ValueError("Choose offline or live explicitly")
    run_id = uuid.uuid4().hex
    folder = run_folder(run_id, artifacts)
    folder.mkdir(parents=True, exist_ok=False)
    store = ShopStore(artifacts / "state.sqlite")
    try:
        telemetry = Telemetry(
            folder,
            run_id,
            config.phoenix_endpoint if config.export_traces else None,
            capture_content=True,
        )
        store.start(run_id, case.case_id)
    except BaseException:
        store.close()
        raise
    inference = Inference(config, telemetry)
    gateway = ToolGateway(
        store.session(run_id), {"get_product"}, config.max_tool_calls, telemetry
    )
    result = dict(
        run_id=run_id,
        case_id=case.case_id,
        mode=mode,
        evidence_kind="authored_control" if mode == "offline" else "live",
        created_at_utc=datetime.now(UTC).isoformat(),
        execution_status="completed",
        verdict="needs_review",
        output=None,
        explanation="",
        source_facts={},
        checks=[],
        attempts=[],
        trace_id="",
        telemetry={},
        provenance={},
        usage={},
    )
    reply = None
    root_span_id = ""
    try:
        with telemetry.span(
            "evaluation.run", case_id=case.case_id, mode=mode
        ) as root_span:
            result["trace_id"] = f"{root_span.get_span_context().trace_id:032x}"
            root_span_id = f"{root_span.get_span_context().span_id:016x}"
            with telemetry.span("application.price"):
                product = gateway.call("get_product", {"product_id": case.product_id})
                result["source_facts"] = product
                if mode == "offline":
                    answer = case.controlled_answer
                else:
                    # Only authoritative public facts and the request reach inference.
                    messages = [
                        {
                            "role": "system",
                            "content": "Answer briefly using only this catalog record. "
                            "Do not invent facts. " + json.dumps(product),
                        },
                        {"role": "user", "content": case.question},
                    ]
                    reply = inference.generate(messages)
                    answer = reply.text
                result["output"] = answer
            with telemetry.span("grading.catalog", "EVALUATOR") as grader_span:
                if reply is not None and reply.finish_reason != "stop":
                    result.update(
                        verdict="needs_review",
                        explanation="Output may be incomplete; review before grading.",
                    )
                else:
                    check = check_catalog(
                        answer, Product.model_validate(product), "price"
                    ).model_dump()
                    result["checks"] = [check]
                    result.update(
                        verdict="needs_review"
                        if check["status"] == "review"
                        else check["status"],
                        explanation=check["reason"],
                    )
                grader_span.set_attribute("evaluation.verdict", result["verdict"])
            telemetry.log("evaluation.completed", verdict=result["verdict"])
    except KeyboardInterrupt:
        result.update(
            execution_status="cancelled",
            verdict="execution_error",
            explanation="Interrupted; no automatic action resume.",
        )
    except Exception as exc:
        result.update(
            execution_status="execution_error",
            verdict="execution_error",
            explanation=str(exc)
            if type(exc).__name__ == "ProviderError"
            else type(exc).__name__,
        )
    finally:
        result["telemetry"] = telemetry.close()
    try:
        if result["telemetry"]["status"] == "exported":
            result["telemetry"].update(
                telemetry.annotate(
                    config.phoenix_endpoint,
                    root_span_id,
                    result["verdict"],
                    result["explanation"],
                )
            )
        else:
            result["telemetry"]["annotation_status"] = "not_submitted"
        completed = [row for row in inference.attempts if row["status"] == "completed"]
        result["attempts"] = inference.attempts
        result["usage"] = {
            "model_calls": len(inference.attempts),
            "tool_calls": gateway.calls,
            "inference_elapsed_ms": sum(
                row["elapsed_ms"] for row in inference.attempts
            ),
            "prompt_tokens": sum(row["prompt_tokens"] for row in completed)
            if completed and all(row["prompt_tokens"] is not None for row in completed)
            else None,
            "completion_tokens": sum(row["completion_tokens"] for row in completed)
            if completed
            and all(row["completion_tokens"] is not None for row in completed)
            else None,
            "usage_incomplete": any(
                row["status"] != "completed" for row in inference.attempts
            )
            or any(
                row.get("prompt_tokens") is None or row.get("completion_tokens") is None
                for row in completed
            ),
            "cost": None,
            "cost_note": "Not measured; never assume zero cost",
        }
        sources = [
            *sorted((ROOT / "advanced/src").rglob("*.py")),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / "intro/src/ai_quality_evals/evaluators/catalog.py",
            FIXTURE,
        ]
        result["provenance"] = {
            "application_version": "advanced-foundation-0.1.0",
            "milestone": "A1",
            "case_sha256": sha256(case.model_dump_json().encode()).hexdigest(),
            "config": config.model_dump(),
            "provider": inference.metadata,
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            "packages": {p: version(p) for p in ("pydantic", "opentelemetry-sdk")},
            "git": git_provenance(ROOT),
            "full_module_validated": False,
            "sampling": {"temperature": 0, "seed": 42, "max_tokens": config.max_tokens},
            "framework_comparison": None,
            "tool_selection": "host_preconfigured_public_lookup",
        }
        record = RunResult.model_validate(redact(result))
        atomic_json(folder / "result.json", record.model_dump())
        atomic_json(
            folder / "metrics.json",
            {
                "run_id": run_id,
                "verdict": record.verdict,
                "usage": record.usage,
                "telemetry": record.telemetry,
            },
        )
        store.finish(run_id, result["execution_status"])
        return record
    except BaseException:
        store.finish(run_id, "artifact_error")
        raise
    finally:
        store.close()


def read_result(identifier: str, *, artifacts: Path = ARTIFACTS) -> RunResult:
    if identifier == "latest":
        paths = sorted(
            artifacts.glob("*/result.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not paths:
            raise ValueError("No saved Advanced runs")
        path = paths[0]
    else:
        path = run_folder(identifier, artifacts) / "result.json"
    return RunResult.model_validate_json(path.read_bytes())
