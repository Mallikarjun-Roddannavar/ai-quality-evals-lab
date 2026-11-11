"""Real OpenTelemetry spans, durable local traces and optional local OTLP export."""

import json
import threading
from contextlib import contextmanager
from pathlib import Path

from ai_quality_evals_advanced.security import redact


def record_evaluation(telemetry, name: str, result: dict):
    """Record synthetic case and check snapshots for the learning examples.

    These spans are created after checks finish: their duration measures recording,
    not the original evaluation. The saved report holds the detailed evidence.
    """
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"cases", "models", "variants"}
    }
    with telemetry.span(
        "evaluation.result",
        "EVALUATOR",
        **{
            "evaluation.name": name,
            "evaluation.snapshot": True,
            "output.value": json.dumps(redact(summary), ensure_ascii=False),
            "output.mime_type": "application/json",
        },
    ):
        pass
    cases = result.get("cases")
    if isinstance(cases, list):
        for index, case in enumerate(cases, start=1):
            if not isinstance(case, dict):
                continue
            attributes = {"case.index": index, "evaluation.snapshot": True}
            attributes["output.value"] = json.dumps(redact(case), ensure_ascii=False)
            attributes["output.mime_type"] = "application/json"
            if name == "foundations":
                attributes["check.name"] = "exact and normalized matching"
                attributes["check.exact"] = case["exact"]
                attributes["check.normalized"] = case["normalized"]
                attributes["case.expected_correct"] = case["label"]
                attributes["case.id"] = case["id"]
            elif name == "skills":
                attributes["check.name"] = "skill selection"
                attributes["check.correct"] = case["correct"]
                selected = case.get("selected") or {}
                allowed = {"cancel-owned-order", "return-owned-order"}
                expected = case.get("expected")
                selected_name = selected.get("name")
                attributes["skill.expected"] = (
                    expected if expected in allowed else "none"
                )
                attributes["skill.selected"] = (
                    selected_name if selected_name in allowed else "none"
                )
            elif name == "retrieval":
                attributes["check.name"] = "document-ID retrieval"
                for key in (
                    "python_recall",
                    "ragas_id_recall",
                    "mrr",
                    "precision",
                    "ndcg",
                ):
                    value = case.get("metrics", {}).get(key)
                    if isinstance(value, (int, float)):
                        attributes["retrieval." + key] = float(value)
            elif name == "judges":
                attributes["check.name"] = "judge support and faithfulness"
                python_judge = case.get("python", {})
                deepeval = case.get("deepeval", {})
                for key in ("supported", "complete"):
                    if isinstance(python_judge.get(key), bool):
                        attributes["judge." + key] = python_judge[key]
                if isinstance(deepeval.get("score"), (int, float)):
                    attributes["judge.faithfulness"] = float(deepeval["score"])
                if python_judge.get("execution_error") or deepeval.get(
                    "execution_error"
                ):
                    attributes["judge.execution_error"] = True
            else:
                attributes["check.name"] = name + " case"
                if isinstance(case.get("outcome_pass"), bool):
                    attributes["check.outcome_pass"] = case["outcome_pass"]
                if case.get("verdict") in {
                    "pass",
                    "fail",
                    "needs_review",
                    "execution_error",
                }:
                    attributes["check.verdict"] = case["verdict"]
                if isinstance(case.get("retrieval_hit"), bool):
                    attributes["check.retrieval_hit"] = case["retrieval_hit"]
                check = case.get("check") or {}
                if isinstance(check, dict) and check.get("status") in {
                    "pass",
                    "fail",
                    "review",
                }:
                    attributes["check.status"] = check["status"]
                if case.get("execution_error"):
                    attributes["case.execution_error"] = True
            span_name = f"evaluation.case.{index:02d}"
            if name == "foundations":
                span_name = "evaluation.case." + case["id"]
            with telemetry.span(span_name, "EVALUATOR", **attributes):
                pass
    variants = result.get("variants")
    if name in {"chat", "rag"} and isinstance(variants, list):
        for index, variant in enumerate(variants, start=1):
            attributes = {
                "variant.index": index,
                "check.name": "conversation review"
                if name == "chat"
                else "citation IDs",
                "evaluation.snapshot": True,
                "output.value": json.dumps(redact(variant), ensure_ascii=False),
                "output.mime_type": "application/json",
            }
            if isinstance(variant.get("citation_ids_valid"), bool):
                attributes["check.citation_ids_valid"] = variant["citation_ids_valid"]
                attributes["check.support"] = "needs_review"
            with telemetry.span("evaluation.variant", "EVALUATOR", **attributes):
                pass
    if name == "mcp" and isinstance(result.get("matched"), list):
        for index, matched in enumerate(result["matched"], start=1):
            with telemetry.span(
                "evaluation.protocol",
                "EVALUATOR",
                **{
                    "case.index": index,
                    "check.name": "direct and MCP result agreement",
                    "check.matched": matched,
                    "evaluation.snapshot": True,
                },
            ):
                pass
    if name == "benchmark" and isinstance(result.get("models"), list):
        for index, model in enumerate(result["models"], start=1):
            sample_count = 0
            for log in model.get("inspect", []):
                sample_count += len(log.get("samples", []))
            with telemetry.span(
                "evaluation.benchmark",
                "EVALUATOR",
                **{
                    "variant.index": index,
                    "check.name": "Inspect benchmark and Python price checks",
                    "check.samples": sample_count,
                    "evaluation.snapshot": True,
                },
            ):
                pass
            for log in model.get("inspect", []):
                for sample in log.get("samples", []):
                    with telemetry.span(
                        "evaluation.benchmark.case",
                        "EVALUATOR",
                        **{
                            "model.requested": model.get("model", "unknown"),
                            "case.id": str(sample.get("id", "unknown")),
                            "output.value": json.dumps(
                                redact(sample), ensure_ascii=False
                            ),
                            "output.mime_type": "application/json",
                            "evaluation.snapshot": True,
                        },
                    ):
                        pass
    if name == "regression":
        for variant in result.get("variants", []):
            with telemetry.span(
                "evaluation.regression",
                "EVALUATOR",
                **{
                    "check.name": "controlled regression gate",
                    "check.variant": variant["variant"],
                    "check.gate_pass": variant["gate_pass"],
                    "evaluation.snapshot": True,
                },
            ) as span:
                for key, value in variant["checks"].items():
                    span.set_attribute("check." + key, value)
    if name in {"foundations", "statistics"}:
        for metric, counts in result.get("counts", {}).items():
            with telemetry.span(
                "evaluation.summary",
                "EVALUATOR",
                **{
                    "check.name": metric + " matching",
                    "check.matches": counts["passes"],
                    "check.total": counts["total"],
                    "check.false_passes": len(counts["false_pass_ids"]),
                    "check.false_failures": len(counts["false_failure_ids"]),
                    "evaluation.snapshot": True,
                },
            ):
                pass


class Telemetry:
    def __init__(
        self,
        folder: Path,
        run_id: str,
        endpoint: str | None,
        *,
        capture_content: bool = False,
    ):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )

        self.folder, self.run_id = folder, run_id
        self.lock = threading.Lock()
        self.exported, self.failures = 0, 0
        self.remote = endpoint is not None
        self.capture_content = capture_content
        self.endpoint = endpoint
        owner = self

        class LocalExporter(SpanExporter):
            def export(self, spans):
                for span in spans:
                    owner.write(
                        "traces.jsonl",
                        {
                            "name": span.name,
                            "trace_id": f"{span.context.trace_id:032x}",
                            "span_id": f"{span.context.span_id:016x}",
                            "parent_span_id": f"{span.parent.span_id:016x}"
                            if span.parent
                            else None,
                            "start_time_ns": span.start_time,
                            "end_time_ns": span.end_time,
                            "status": span.status.status_code.name,
                            "attributes": dict(span.attributes or {}),
                        },
                    )
                return SpanExportResult.SUCCESS

        class CountingExporter(SpanExporter):
            def __init__(self):
                import requests

                class LocalSession(requests.Session):
                    def request(self, *args, **kwargs):
                        kwargs["allow_redirects"] = False
                        return super().request(*args, **kwargs)

                session = LocalSession()
                session.trust_env = False
                self.wrapped = OTLPSpanExporter(
                    endpoint=endpoint, timeout=1, headers={}, session=session
                )

            def export(self, spans):
                try:
                    status = self.wrapped.export(spans)
                except Exception:
                    status = SpanExportResult.FAILURE
                with owner.lock:
                    if status == SpanExportResult.SUCCESS:
                        owner.exported += len(spans)
                    else:
                        owner.failures += len(spans)
                return status

            def shutdown(self):
                self.wrapped.shutdown()

        self.provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": "demoshop-advanced",
                    "openinference.project.name": "demoshop-advanced",
                }
            )
        )
        self.provider.add_span_processor(SimpleSpanProcessor(LocalExporter()))
        if endpoint:
            self.provider.add_span_processor(
                BatchSpanProcessor(
                    CountingExporter(),
                    max_queue_size=256,
                    max_export_batch_size=64,
                    schedule_delay_millis=100,
                    export_timeout_millis=1000,
                )
            )
        self.tracer = self.provider.get_tracer("demoshop-advanced", "0.1.0")

    def write(self, name: str, value: dict):
        with self.lock:
            with (self.folder / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(redact(value)) + "\n")

    @contextmanager
    def span(self, name: str, kind: str = "CHAIN", **attributes):
        from opentelemetry.trace import StatusCode

        with self.tracer.start_as_current_span(
            name,
            record_exception=False,
            set_status_on_exception=False,
            attributes=redact(
                {"run.id": self.run_id, "openinference.span.kind": kind, **attributes}
            ),
        ) as span:
            try:
                yield span
            except BaseException as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(StatusCode.ERROR)
                raise

    def log(self, event: str, **metadata):
        from opentelemetry.trace import get_current_span

        context = get_current_span().get_span_context()
        self.write(
            "events.jsonl",
            {
                "run_id": self.run_id,
                "event": event,
                "trace_id": f"{context.trace_id:032x}",
                "span_id": f"{context.span_id:016x}",
                **metadata,
            },
        )

    def close(self) -> dict:
        flushed = self.provider.force_flush(timeout_millis=3000)
        self.provider.shutdown()
        return {
            "status": "local_only"
            if not self.remote
            else "exported"
            if self.exported and not self.failures and flushed
            else "degraded",
            "exported_spans": self.exported,
            "failed_spans": self.failures,
            "local_traces": "traces.jsonl",
            "raw_content_capture": self.capture_content,
        }

    def annotate(
        self,
        endpoint: str,
        span_id: str,
        verdict: str,
        explanation: str,
        criterion: str = "catalog_price",
        kind: str = "CODE",
    ) -> dict:
        """Submit a labeled machine annotation; backend outages remain explicit."""
        import http.client
        from urllib.parse import urlsplit

        annotation = redact(
            {
                "name": criterion,
                "annotator_kind": kind,
                "span_id": span_id,
                "result": {
                    "label": verdict,
                    "explanation": explanation,
                    **(
                        {"score": 1.0 if verdict == "pass" else 0.0}
                        if verdict in {"pass", "fail"}
                        else {}
                    ),
                },
                "metadata": {
                    "run_id": self.run_id,
                    "criterion": "bounded_catalog_price",
                },
            }
        )
        self.write("annotations.jsonl", annotation)
        url = urlsplit(endpoint)
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=1)
        try:
            conn.request(
                "POST",
                "/v1/span_annotations",
                json.dumps({"data": [annotation]}),
                {"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            return {
                "annotation_status": "submitted"
                if response.status in {200, 201, 202, 204}
                else "failed"
            }
        except (OSError, http.client.HTTPException):
            return {"annotation_status": "failed"}
        finally:
            conn.close()
