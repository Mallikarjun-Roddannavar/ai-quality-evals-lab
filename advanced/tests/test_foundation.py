"""A1 real SDK tracing and durable state; inference is explicitly doubled."""

import gzip
import json
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from pydantic import ValidationError

from ai_quality_evals_advanced import cli, providers, runtime
from ai_quality_evals_advanced.contracts import Settings, run_folder
from ai_quality_evals_advanced.domain import ShopStore, ToolGateway, snapshot
from ai_quality_evals_advanced.observability import phoenix_environment
from ai_quality_evals_advanced.providers import Inference, ProviderError
from ai_quality_evals_advanced.providers import request as allowed_request
from ai_quality_evals_advanced.security import redact
from ai_quality_evals_advanced.telemetry import Telemetry


@pytest.fixture(autouse=True)
def no_live_inference(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    def deny(*args, **kwargs):
        raise AssertionError("Offline tests must not contact a model service")

    monkeypatch.setattr(providers, "request", deny)
    monkeypatch.setattr(cli, "request", deny)


@pytest.fixture
def store(tmp_path):
    database = ShopStore(tmp_path / "state.sqlite")
    database.start("one", "test-case")
    yield database
    database.close()


def test_authorization_idempotency_and_trial_isolation(store):
    owner = store.session("one", "C001")
    for operation in (owner.get_order, owner.cancel_order, owner.request_return):
        with pytest.raises(PermissionError):
            operation("DS-1003")
    with pytest.raises(PermissionError):
        owner.cancel_order("DS-1002")
    assert owner.cancel_order("DS-1001")["changed"]
    assert not owner.cancel_order("DS-1001")["changed"]
    store.start("two", "test-case")
    assert store.session("two").get_order("DS-1001")["status"] == "awaiting shipment"
    assert owner.get_order("DS-1001")["status"] == "cancelled"
    with pytest.raises(ValueError):
        owner.get_order("DS-1001' OR 1=1 --")
    assert (
        store.db.execute(
            "SELECT count(*) FROM audit WHERE outcome='denied'"
        ).fetchone()[0]
        >= 4
    )


def test_return_request_is_durable_and_idempotent(store):
    owner = store.session("one", "C002")
    first = owner.request_return("DS-1003")
    again = owner.request_return("DS-1003")
    assert first["changed"] and not again["changed"]
    assert first["return_id"] == again["return_id"]
    assert owner.get_order("DS-1003")["status"] == "return requested"


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, True), (14 * 86400, True), (14 * 86400 + 1, False), (-1, False)],
)
def test_return_utc_boundaries(store, seconds, expected):
    order = store.session("one").get_order("DS-1001")
    order["status"] = "delivered"
    order["delivered_at_utc"] = (
        datetime.fromisoformat(snapshot()["as_of_utc"]) - timedelta(seconds=seconds)
    ).isoformat()
    store.db.execute(
        "UPDATE orders SET payload=? WHERE run_id=? AND id=?",
        (json.dumps(order), "one", "DS-1001"),
    )
    assert (
        store.session("one").check_return_eligibility("DS-1001")["eligible"] is expected
    )


def test_gateway_rejects_forged_identity_write_capability_and_budget(store):
    gateway = ToolGateway(store.session("one"), {"get_order"}, limit=2)
    with pytest.raises(ValueError):
        gateway.call("get_order", {"order_id": "DS-1003", "actor": "C002"})
    with pytest.raises(PermissionError):
        gateway.call("cancel_order", {"order_id": "DS-1001"})
    with pytest.raises(RuntimeError, match="budget"):
        gateway.call("get_order", {"order_id": "DS-1001"})
    assert store.session("one").get_order("DS-1001")["status"] == "awaiting shipment"


def test_offline_evidence_traces_and_replay_never_infer(tmp_path):
    record = runtime.execute(
        runtime.load_case(),
        "offline",
        Settings(export_traces=False),
        artifacts=tmp_path,
    )
    assert record.verdict == "pass" and record.evidence_kind == "authored_control"
    assert record.usage["model_calls"] == 0
    folder = run_folder(record.run_id, tmp_path)
    before = (folder / "result.json").read_bytes()
    assert runtime.read_result(record.run_id, artifacts=tmp_path) == record
    assert before == (folder / "result.json").read_bytes()
    spans = [
        json.loads(line) for line in (folder / "traces.jsonl").read_text().splitlines()
    ]
    assert {s["name"] for s in spans} == {
        "evaluation.run",
        "application.price",
        "grading.catalog",
        "tool.get_product",
    }
    assert {s["trace_id"] for s in spans} == {record.trace_id}
    parent = next(s for s in spans if s["name"] == "evaluation.run")
    app = next(s for s in spans if s["name"] == "application.price")
    tool = next(s for s in spans if s["name"] == "tool.get_product")
    assert tool["parent_span_id"] == app["span_id"]
    assert app["parent_span_id"] == parent["span_id"]
    assert tool["attributes"]["security.decision"] == "allowed"
    assert "800 INR" not in (folder / "traces.jsonl").read_text()


def test_wrong_price_is_quality_failure_not_runner_error(tmp_path):
    case = runtime.load_case().model_copy(update={"controlled_answer": "900 INR"})
    record = runtime.execute(
        case, "offline", Settings(export_traces=False), artifacts=tmp_path
    )
    assert record.execution_status == "completed" and record.verdict == "fail"
    assert record.explanation == "price_mismatch"
    assert record.usage["model_calls"] == 0


def fake_transport(failures=0, text="The backpack costs 800 INR."):
    attempts = []

    def transport(url, payload, timeout, *args):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "gemma4:e2b", "digest": "offline-double"}]}
        if url.endswith("/api/show"):
            return {"capabilities": ["completion", "thinking"]}
        if url.endswith("/api/version"):
            return {"version": "offline-double"}
        attempts.append(payload)
        assert payload["think"] is False
        if len(attempts) <= failures:
            raise ProviderError("http_503", True)
        return {
            "done": True,
            "done_reason": "stop",
            "model": "gemma4:e2b",
            "message": {"content": text},
            "prompt_eval_count": 20,
            "eval_count": 8,
        }

    return transport, attempts


def test_live_path_records_retry_error_span_and_partial_usage(tmp_path, monkeypatch):
    fake, attempts = fake_transport(1)
    monkeypatch.setattr(providers, "request", fake)
    record = runtime.execute(
        runtime.load_case(), "live", Settings(export_traces=False), artifacts=tmp_path
    )
    assert record.verdict == "pass" and len(attempts) == 2
    assert record.usage["model_calls"] == 2 and record.usage["usage_incomplete"]
    assert all(row["elapsed_ms"] > 0 for row in record.attempts)
    spans = [
        json.loads(s)
        for s in (run_folder(record.run_id, tmp_path) / "traces.jsonl")
        .read_text()
        .splitlines()
    ]
    calls = [s for s in spans if s["name"] == "model.generate"]
    assert len(calls) == 2 and calls[0]["status"] == "ERROR"


def test_retry_cannot_exceed_actual_call_budget(tmp_path, monkeypatch):
    fake, attempts = fake_transport(3)
    monkeypatch.setattr(providers, "request", fake)
    record = runtime.execute(
        runtime.load_case(),
        "live",
        Settings(export_traces=False, max_model_calls=1),
        artifacts=tmp_path,
    )
    assert record.execution_status == "execution_error"
    assert len(attempts) == record.usage["model_calls"] == 1


def test_deadline_prevents_request(tmp_path):
    telemetry = Telemetry(tmp_path, "test", None)
    inference = Inference(Settings(), telemetry)
    inference.deadline = time.monotonic() - 1
    with pytest.raises(ProviderError, match="deadline"):
        inference.generate([])
    assert not inference.attempts
    telemetry.close()


def test_missing_model_is_saved_error_without_fallback(tmp_path, monkeypatch):
    calls = []

    def tags(url, payload, timeout):
        calls.append(url)
        return {"models": []}

    monkeypatch.setattr(providers, "request", tags)
    record = runtime.execute(
        runtime.load_case(), "live", Settings(export_traces=False), artifacts=tmp_path
    )
    assert record.execution_status == "execution_error"
    assert "no_download_attempted" in record.explanation
    assert len(calls) == 1 and record.output is None


def test_secrets_are_redacted_in_saved_result_and_validation_errors(
    tmp_path, monkeypatch
):
    secret = "sk-or-v1-SYNTHETIC_ONLY_TEST_KEY"
    fake, _ = fake_transport(text=secret)
    monkeypatch.setattr(providers, "request", fake)
    record = runtime.execute(
        runtime.load_case(), "live", Settings(export_traces=False), artifacts=tmp_path
    )
    assert record.output == "[REDACTED]"
    for p in run_folder(record.run_id, tmp_path).iterdir():
        assert secret not in p.read_text()
    assert redact({"Authorization": "value", "message": "Bearer synthetic"}) == {
        "Authorization": "[REDACTED]",
        "message": "[REDACTED]",
    }
    with pytest.raises(ValidationError) as error:
        Settings.model_validate({"api_key": "UNPREFIXED_SYNTHETIC_SECRET"})
    assert "UNPREFIXED_SYNTHETIC_SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://user:pass@localhost:6006",
        "http://example.com/v1/traces",
    ],
)
def test_remote_telemetry_and_local_credentials_rejected(url):
    with pytest.raises(ValidationError):
        Settings(phoenix_endpoint=url)


def test_artifact_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        runtime.read_result("../intro", artifacts=tmp_path)


def test_atomic_failure_preserves_committed_report(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    runtime.atomic_json(path, {"version": 1})
    before = path.read_bytes()

    def fail(*args):
        raise OSError("Synthetic replacement failure")

    monkeypatch.setattr(runtime.os, "replace", fail)
    with pytest.raises(OSError):
        runtime.atomic_json(path, {"version": 2})
    assert path.read_bytes() == before


def test_actual_otlp_export_and_trace_correlation(tmp_path):
    bodies = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path in {"/v1/traces", "/v1/span_annotations"}
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/v1/traces":
                bodies.append(
                    gzip.decompress(body)
                    if self.headers.get("Content-Encoding") == "gzip"
                    else body
                )
            else:
                assert json.loads(body)["data"][0]["annotator_kind"] == "CODE"
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = Settings(
            phoenix_endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces"
        )
        record = runtime.execute(
            runtime.load_case(), "offline", config, artifacts=tmp_path
        )
        assert record.telemetry["status"] == "exported" and bodies
        assert record.telemetry["annotation_status"] == "submitted"
        spans = [
            span
            for body in bodies
            for resource in ExportTraceServiceRequest.FromString(body).resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        ]
        assert len(spans) == 4
        assert {s.trace_id.hex() for s in spans} == {record.trace_id}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_export_outage_preserves_local_evidence(tmp_path):
    # Port 1 is deliberately closed on the local test machine; no model call.
    result = runtime.execute(
        runtime.load_case(),
        "offline",
        Settings(phoenix_endpoint="http://127.0.0.1:1/v1/traces"),
        artifacts=tmp_path,
    )
    assert result.verdict == "pass" and result.telemetry["status"] == "degraded"
    assert result.telemetry["failed_spans"] > 0
    assert runtime.read_result(result.run_id, artifacts=tmp_path) == result


def test_doctor_offline_and_explicit_execution_mode(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evals", "doctor", "--offline"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["docker_required"] is False
    monkeypatch.setattr("sys.argv", ["evals", "run"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_native_phoenix_environment_is_local_and_does_not_inherit_keys(
    tmp_path, monkeypatch
):
    from ai_quality_evals_advanced import observability

    monkeypatch.setattr(observability, "ARTIFACTS", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "SYNTHETIC_DO_NOT_INHERIT")
    monkeypatch.setenv("LANGSMITH_API_KEY", "SYNTHETIC_DO_NOT_INHERIT")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://example.com")
    env = phoenix_environment(Settings())
    assert "OPENROUTER_API_KEY" not in env and "LANGSMITH_API_KEY" not in env
    assert "PHOENIX_COLLECTOR_ENDPOINT" not in env
    assert env["PHOENIX_HOST"] == "127.0.0.1"
    assert env["PHOENIX_WORKING_DIR"] == str(tmp_path / "phoenix")
    assert env["PHOENIX_ALLOW_EXTERNAL_RESOURCES"] == "false"


def test_interruption_is_durable_and_never_resumes_actions(tmp_path, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "check_catalog", interrupt)
    record = runtime.execute(
        runtime.load_case(),
        "offline",
        Settings(export_traces=False),
        artifacts=tmp_path,
    )
    assert record.execution_status == "cancelled"
    assert (
        runtime.read_result(record.run_id, artifacts=tmp_path).execution_status
        == "cancelled"
    )


def test_failed_report_write_marks_run_artifact_error(tmp_path, monkeypatch):
    import sqlite3

    def failed_write(*args):
        raise OSError("Synthetic disk failure")

    monkeypatch.setattr(runtime, "atomic_json", failed_write)
    with pytest.raises(OSError):
        runtime.execute(
            runtime.load_case(),
            "offline",
            Settings(export_traces=False),
            artifacts=tmp_path,
        )
    with sqlite3.connect(tmp_path / "state.sqlite") as db:
        assert db.execute("SELECT status FROM runs").fetchone()[0] == "artifact_error"


def test_http_transport_rejects_unapproved_destination_and_never_redirects(monkeypatch):
    seen = []

    class Connection:
        def __init__(self, host, port, timeout):
            seen.append(host)

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return type("Redirect", (), {"status": 302})()

        def close(self):
            pass

    monkeypatch.setattr(providers.http.client, "HTTPConnection", Connection)
    with pytest.raises(ProviderError, match="destination"):
        allowed_request("https://example.com/chat", {}, 1)
    with pytest.raises(ProviderError, match="302"):
        allowed_request("http://127.0.0.1:11434/api/chat", {}, 1)
    assert seen == ["127.0.0.1"]


def test_denied_action_has_error_trace_and_unchanged_state(store, tmp_path):
    telemetry = Telemetry(tmp_path, "denial", None)
    gateway = ToolGateway(store.session("one"), {"cancel_order"}, telemetry=telemetry)
    with telemetry.span("security.case"):
        with pytest.raises(PermissionError):
            gateway.call("cancel_order", {"order_id": "DS-1003"})
    telemetry.close()
    spans = [
        json.loads(s) for s in (tmp_path / "traces.jsonl").read_text().splitlines()
    ]
    denied = next(s for s in spans if s["name"] == "tool.cancel_order")
    assert denied["attributes"]["security.decision"] == "denied"
    assert denied["status"] == "ERROR"
    assert store.session("one", "C002").get_order("DS-1003")["status"] == "delivered"
