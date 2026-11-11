"""Real protocol, retrieval scoring, procedure routing and native call controls."""

import json
import uuid
from types import SimpleNamespace

import pytest

from ai_quality_evals_advanced import applications
from ai_quality_evals_advanced.contracts import Reply
from ai_quality_evals_advanced.domain import ShopStore, ToolGateway
from ai_quality_evals_advanced.frameworks import retrieval_metrics
from ai_quality_evals_advanced.scenarios import protocol_comparison, skill_probes
from ai_quality_evals_advanced.telemetry import Telemetry


@pytest.mark.parametrize("variant", ["python", "langchain"])
def test_chat_logs_each_call_and_prior_response(tmp_path, variant):
    telemetry = Telemetry(tmp_path, "test", None, capture_content=True)
    replies = iter(
        [Reply(text="Backpack", model="test"), Reply(text="800 INR", model="test")]
    )
    inference = SimpleNamespace(telemetry=telemetry, generate=lambda *a: next(replies))
    try:
        applications.chat(inference, ["Backpack?", "Its price?"], variant)
    finally:
        telemetry.close()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "chat.call.started",
        "chat.call.completed",
        "chat.call.started",
        "chat.call.completed",
    ]
    assert events[0]["messages"][-1]["content"] == "Backpack?"
    assert events[0]["variant"] == variant
    assert events[2]["messages"][-2:] == [
        {"role": "assistant", "content": "Backpack"},
        {"role": "user", "content": "Its price?"},
    ]
    assert events[3]["response"] == "800 INR"
    spans = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert spans[-1]["attributes"]["output.value"] == "800 INR"


def test_real_mcp_state_and_denial(tmp_path):
    id = uuid.uuid4().hex
    folder = tmp_path / id
    folder.mkdir()
    telemetry = Telemetry(folder, id, None)
    try:
        result = protocol_comparison(folder, id, telemetry)
        assert result["matched"] == [True, True, True]
        assert result["mcp"]["transport"] == "real_stdio"
        assert result["mcp"]["results"][-1]["is_error"]
        server_traces = [
            json.loads(line)
            for line in (folder / "mcp-server/traces.jsonl").read_text().splitlines()
        ]
        client_traces = [
            json.loads(line)
            for line in (folder / "traces.jsonl").read_text().splitlines()
        ]
        assert {s["trace_id"] for s in server_traces} <= {
            s["trace_id"] for s in client_traces
        }
    finally:
        telemetry.close()


def test_rag_does_not_treat_tool_markup_as_citation(tmp_path):
    telemetry = Telemetry(tmp_path, "test", None)
    inference = SimpleNamespace(
        telemetry=telemetry,
        generate=lambda *args: Reply(
            text="<|tool_call_start|>[find(source='[returns]')]<|tool_call_end|>",
            model="test",
        ),
    )
    try:
        result = applications.rag(inference, "How many days for returns?")
    finally:
        telemetry.close()
    assert result["citation_ids_valid"] is False
    assert result["answer_review_reason"] == "tool_call_markup_in_answer"
    assert result["support_verdict"] == "needs_review"


def test_routing_paths_and_retrieval_failures(monkeypatch, tmp_path):
    assert skill_probes()["routing_accuracy"] == 1
    with pytest.raises(ValueError):
        applications.load_skill("../../untrusted")
    monkeypatch.setattr(applications, "SKILLS", tmp_path)
    with pytest.raises(FileNotFoundError):
        applications.load_skill("cancel-owned-order")
    scores = retrieval_metrics(["irrelevant", "returns"], ["returns"])
    assert scores["python_recall"] == scores["ragas_id_recall"] == 1
    assert scores["mrr"] == scores["precision"] == 0.5
    assert retrieval_metrics([], ["returns"])["ragas_id_recall"] == 0
    assert all(d["active"] for d in applications.retrieve("returns"))


def test_native_tool_control_cannot_impersonate(tmp_path):
    store = ShopStore(tmp_path / "state.sqlite")
    store.start("test", "native-control")
    gateway = ToolGateway(store.session("test"), {"get_order"})
    replies = iter(
        [
            Reply(
                text="",
                model="control",
                tool_calls=[
                    {
                        "function": {
                            "name": "get_order",
                            "arguments": {"order_id": "DS-1003", "actor": "C002"},
                        }
                    }
                ],
            ),
            Reply(text="Access denied", model="control"),
        ]
    )
    inference = SimpleNamespace(generate=lambda *a, **kw: next(replies))
    try:
        result = applications.tool_loop(
            inference, gateway, "Get another customer's order"
        )
        assert result["actions"][0]["result"] == {"error": "ValueError"}
        assert (
            store.session("test").get_order("DS-1001")["status"] == "awaiting shipment"
        )
    finally:
        store.close()
