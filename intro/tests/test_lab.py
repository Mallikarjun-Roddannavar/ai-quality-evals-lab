"""Offline coverage of every starter demo, real MCP and browser rerun behavior."""

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from ai_quality_evals import lab
from ai_quality_evals import lab_evals as demos
from ai_quality_evals.lab_systems import (
    SKILL,
    Runtime,
    Shop,
    load_skill,
    run_agent,
    skill_for,
)
from ai_quality_evals.providers import ollama, openrouter


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    calls = []
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OCI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_EVAL_PROVIDER", "ollama")
    monkeypatch.setattr(openrouter, "load_dotenv", lambda path=None: None)
    monkeypatch.setattr(lab, "load_dotenv", lambda: None)

    def deny_network(*args, **kwargs):
        raise AssertionError("Offline tests must not call a model service")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    # Windows asyncio builds its internal wakeup pipe with loopback socketpair.
    # Permit that internal transport, while blocking model HTTP and external hosts.
    original_connect = socket.socket.connect

    def internal_only(sock, address):
        if (
            isinstance(address, tuple)
            and address[0] in {"127.0.0.1", "::1"}
            and address[1] != 11434
        ):
            return original_connect(sock, address)
        return deny_network(sock, address)

    monkeypatch.setattr(socket.socket, "connect", internal_only)
    monkeypatch.setattr(ollama, "urlopen", deny_network)
    monkeypatch.setattr(demos, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(
        ollama,
        "installed_model",
        lambda url, model: {
            "name": model,
            "digest": "offline-test-double",
            "ollama_version": "test",
        },
    )

    def complete(url, model, messages, **kwargs):
        calls.append({"model": model, "messages": messages, **kwargs})
        schema = kwargs.get("response_format")
        text = " ".join(message["content"] for message in messages)
        question = messages[-1]["content"]
        if messages[-1]["role"] == "tool":
            if kwargs.get("tools"):
                return ollama.ToolCompletion(
                    output="The notebook costs 100 INR.",
                    tool_calls=[],
                    model=model,
                    finish_reason="stop",
                    prompt_tokens=10,
                    completion_tokens=8,
                    latency_ms=3.0,
                    raw_response={
                        "message": {
                            "role": "assistant",
                            "content": "The notebook costs 100 INR.",
                        },
                        "test_double": True,
                    },
                )
            return ollama.Completion(
                output="The notebook costs 100 INR.",
                model=model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=8,
                latency_ms=3.0,
                raw_response={"test_double": True},
            )
        if kwargs.get("tools"):
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_product",
                            "arguments": {"product_id": "SKU-NOTEBOOK"},
                        }
                    }
                ],
            }
            return ollama.ToolCompletion(
                output=message["content"],
                tool_calls=message.get("tool_calls", []),
                model=model,
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=8,
                latency_ms=3.0,
                raw_response={"message": message, "test_double": True},
            )
        if schema and "tool" in schema["properties"]:
            if "Select get_product" in text:
                output = {"tool": "get_product", "product_id": "SKU-NOTEBOOK"}
            elif "Actual tool result" in text or "Inspection and reviewer" in text:
                output = {"tool": "cancel_order", "order_id": "DS-1001"}
            else:
                output = {"tool": "get_order", "order_id": "DS-1001"}
            answer = json.dumps(output)
        elif schema:
            enum = schema["properties"]["verdict"]["enum"]
            answer = json.dumps(
                {
                    "verdict": "allowed" if "allowed" in enum else "uncertain",
                    "reason": "Authored unit-test double, not a real judgment",
                }
            )
        elif "remember" in question.lower():
            answer = "I will remember the backpack."
        elif "How much does it cost" in question or "backpack cost" in question:
            answer = "The backpack costs 800 INR."
        elif "notebook cost" in question:
            answer = "The notebook costs 100 INR."
        elif "water bottle" in question:
            answer = "No, the water bottle is out of stock."
        elif "cancel" in question.lower():
            answer = "Cancellation is allowed only before shipment. [cancellation]"
        else:
            answer = "The supplied information does not specify international shipping."
        return ollama.Completion(
            output=answer,
            model=model,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=8,
            latency_ms=3.0,
            raw_response={"message": {"content": answer}, "test_double": True},
        )

    monkeypatch.setattr(ollama, "chat", complete)
    return calls


@pytest.mark.parametrize("topic", [f"{i:02}" for i in range(1, 17)])
def test_each_demo_runs_and_roundtrips(topic, fake_models):
    result = demos.run_demo(topic)
    assert result.status != "execution_error", result.output
    assert result.explanation and result.limitation and result.expected
    assert result.source_facts["products"][0]["price_inr"] == 100
    assert result.provenance["full_module_validated"] is False
    path = demos.save_result(result)
    before = path.read_bytes()
    call_count = len(fake_models)
    replayed = demos.replay(path)
    assert replayed.output == result.output
    assert replayed.evidence_kind == "replay"
    assert path.read_bytes() == before and len(fake_models) == call_count
    assert result.usage["model_calls"] <= (6 if topic == "05" else 4)
    if topic == "09":
        assert result.output["transport"] == "stdio"
        assert not result.output["response"]["is_error"]
        assert result.output["invalid_arguments"]["is_error"]
    if topic == "12":
        handoffs = [e for e in result.trace if e["kind"] == "handoff"]
        assert len(handoffs) == 1
        roles = {e["role"] for e in result.trace if e["kind"] == "model"}
        assert {"Single agent", "Policy reviewer", "Order worker"} <= roles
        assert result.output["single_agent"]["final_order"]["status"] == "cancelled"
        assert result.output["two_roles"]["final_order"]["status"] == "cancelled"
    if topic == "11":
        assert [e["tool"] for e in result.trace if e["kind"] == "tool"] == [
            "get_order",
            "cancel_order",
        ]
    if topic == "08":
        assert result.status == "pass"
        assert result.output["native_message"]["tool_calls"]
        assert fake_models[-1]["messages"][-1]["role"] == "tool"
        assert fake_models[-1]["tools"]
        assert result.checks[0]["returned_fact_correct"] is True
    if topic == "10":
        assert result.output["matching_request"]["path"].endswith("/SKILL.md")
        assert result.checks[0]["skill_body_in_prompt"]
        assert result.output["execution"]["final_order"]["status"] == "cancelled"
        assert any(e["kind"] == "skill_load" for e in result.trace)
        action_schemas = [call["response_format"] for call in fake_models]
        assert action_schemas
        for schema in action_schemas:
            assert schema["required"] == list(schema["properties"])
            assert all(
                "default" not in field for field in schema["properties"].values()
            )


def test_tool_authorization_state_reset_and_idempotency():
    shop = Shop("C001")
    for operation in (shop.get_order, shop.cancel_order):
        with pytest.raises(PermissionError):
            operation("DS-1003")
    with pytest.raises(PermissionError):
        shop.cancel_order("DS-1002")
    assert shop.cancel_order("DS-1001")["changed"] is True
    assert shop.cancel_order("DS-1001")["changed"] is False
    assert Shop().get_order("DS-1001")["status"] == "awaiting shipment"


def test_skill_pass_requires_instructions_in_model_prompt(fake_models, monkeypatch):
    monkeypatch.setattr(
        demos,
        "run_agent",
        lambda runtime, skill: {"final_order": {"status": "cancelled"}},
    )
    result = demos.run_demo("10")
    assert result.checks[0]["skill_body_in_prompt"] is False
    assert result.status == "fail"


def test_security_output_shows_unexpected_access_without_crashing():
    displayed = []
    st = SimpleNamespace(
        subheader=lambda *args: None,
        write=lambda *args: displayed.append(args),
    )
    result = SimpleNamespace(
        topic_id="13",
        status="fail",
        output={
            "cross_customer_read": {"order_id": "DS-1003"},
            "cross_customer_cancel": {"status": "cancelled"},
            "legitimate_read": {"error": "denied"},
            "state_unchanged": False,
        },
    )
    lab.show_output(st, result)
    assert ("Your own order accessible:", False) in displayed
    assert ("Order state unchanged:", False) in displayed


def test_openrouter_runtime_uses_selected_provider_without_ollama(
    fake_models, monkeypatch
):
    def complete(model, messages, **kwargs):
        return ollama.Completion(
            output="The backpack costs 800 INR.",
            model=model,
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=4,
            cost=0.0000465,
            latency_ms=2,
            raw_response={"choices": []},
        )

    monkeypatch.setattr(openrouter, "chat", complete)
    result = demos.run_demo("01", provider="openrouter", model="vendor/model")
    assert result.status != "execution_error"
    assert result.output == "The backpack costs 800 INR."
    assert result.provenance["provider"] == "openrouter"
    assert result.provenance["models"]["vendor/model"]["provider"] == "openrouter"
    assert result.usage["cost"] == 0.0000465
    assert result.usage["cost_currency"] == "USD"
    assert not result.usage["missing_cost_measurements"]
    assert fake_models == []


def test_budgets_stop_an_unproductive_agent(fake_models, monkeypatch):
    monkeypatch.setattr(
        ollama,
        "chat",
        lambda *args, **kwargs: ollama.Completion(
            output='{"tool":"get_order","order_id":"DS-1001"}',
            model="test",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
            raw_response={},
        ),
    )
    runtime = Runtime(max_calls=4, max_tools=3)
    with pytest.raises(RuntimeError, match="Tool-call budget"):
        run_agent(runtime)
    assert runtime.usage()["model_calls"] == 4
    assert runtime.usage()["tool_calls"] == 3
    with pytest.raises(RuntimeError, match="Model-call budget"):
        runtime.infer([])


def test_missing_model_never_silently_replays(fake_models, monkeypatch):
    def missing(*args):
        raise ValueError("Model is not installed; no download was attempted")

    monkeypatch.setattr(ollama, "installed_model", missing)
    result = demos.run_demo("01")
    assert result.status == "execution_error"
    assert "not installed" in result.output["error"]
    assert result.usage["missing_token_measurements"]
    assert not fake_models


def test_thinking_is_disabled_only_for_models_advertising_support(
    fake_models, monkeypatch
):
    monkeypatch.setattr(
        ollama,
        "installed_model",
        lambda *args: {
            "name": "thinking-model",
            "capabilities": ["completion", "thinking"],
        },
    )
    result = demos.run_demo("01")
    assert result.status != "execution_error"
    assert fake_models[0]["thinking"] is False
    assert result.trace[0]["generation_settings"]["think"] is False


def test_human_labels_are_explicit_and_do_not_overwrite_evidence(fake_models):
    result = demos.run_demo("03")
    evidence = demos.save_result(result)
    before = evidence.read_bytes()
    with pytest.raises(ValueError):
        demos.save_human_review(result, "", "correct", "")
    review_path = demos.save_human_review(
        result, "Unit-test reviewer", "uncertain", "Synthetic test submission"
    )
    review = json.loads(review_path.read_text())
    assert review["source"] == "user_entered"
    assert review["independent_agreement_established"] is False
    assert evidence.read_bytes() == before


def test_wilson_hand_cases_and_zero_denominator():
    assert demos.wilson(0, 0) is None
    lower, upper = demos.wilson(3, 5)
    assert lower == pytest.approx(0.230724, abs=0.000001)
    assert upper == pytest.approx(0.882379, abs=0.000001)
    assert demos.wilson(0, 5)[0] == 0
    with pytest.raises(ValueError):
        demos.wilson(6, 5)


def test_regression_cli_exit_status_and_baseline(fake_models, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lab", "--check"])
    assert lab.main() == 0
    monkeypatch.setattr(sys, "argv", ["lab", "--check", "--broken"])
    assert lab.main() == 1
    assert not fake_models


def test_browser_never_calls_models_on_load_rerun_or_replay(fake_models):
    app = AppTest.from_file(str(Path(lab.__file__)), default_timeout=30).run()
    assert not app.exception and not fake_models
    assert len(app.selectbox[0].options) == 16
    app.button[0].click().run()
    assert not app.exception and len(fake_models) == 1
    app.run()
    assert len(fake_models) == 1
    app.text_input[0].input("another-installed-tag").run()
    assert len(fake_models) == 1
    app.button[1].click().run()
    assert not app.exception and len(fake_models) == 1
    assert any("RECORDED REPLAY" in item.value for item in app.caption)


def test_all_topics_render_their_results_in_browser(fake_models):
    app = AppTest.from_file(str(Path(lab.__file__)), default_timeout=30).run()
    for topic_id, *_ in demos.TOPICS:
        app.selectbox[0].set_value(topic_id).run()
        app.button[0].click().run()
        assert not app.exception, (topic_id, app.exception)
        assert any(item.value == "Shop facts" for item in app.subheader)
        assert any(item.value == "What happened" for item in app.subheader)
        assert any("Input / task:" in item.value for item in app.markdown)
        assert any("Expected behavior:" in item.value for item in app.markdown)
        assert any("Check:" in item.value for item in app.markdown)


def test_execution_error_explains_that_no_check_was_completed(fake_models, monkeypatch):
    def fail_call(*args, **kwargs):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(ollama, "chat", fail_call)
    app = AppTest.from_file(str(Path(lab.__file__)), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Model unavailable" in item.value for item in app.markdown)
    assert any("INR 800" in item.value for item in app.markdown)
    assert any("no pass or fail was determined" in item.value for item in app.caption)


def test_human_form_requires_input_and_saves_without_new_inference(fake_models):
    app = AppTest.from_file(str(Path(lab.__file__)), default_timeout=30).run()
    app.selectbox[0].set_value("03").run()
    app.button[0].click().run()
    assert len(fake_models) == 1
    submit = next(b for b in app.button if b.label == "Save my assessment")
    submit.click().run()
    assert app.error and not list((demos.REPORTS / "reviews").glob("*.json"))
    next(w for w in app.text_input if w.label == "Your name").input("UI test reviewer")
    next(w for w in app.selectbox if w.label == "Your label").set_value("correct")
    next(w for w in app.text_area if w.label == "Why?").input(
        "Synthetic UI-test annotation"
    )
    next(b for b in app.button if b.label == "Save my assessment").click().run()
    assert not app.exception and len(fake_models) == 1
    assert len(list((demos.REPORTS / "reviews").glob("*.json"))) == 1


def test_skill_is_loaded_from_disk_and_its_exact_body_reaches_model(
    fake_models, tmp_path, monkeypatch
):
    folder = tmp_path / "cancel-owned-order"
    folder.mkdir()
    path = folder / "SKILL.md"
    path.write_bytes(SKILL.read_bytes() + b"\nUnique file-backed test instruction.\n")
    loaded = load_skill(path)
    monkeypatch.setattr(
        demos,
        "skill_for",
        lambda question: loaded if "cancel" in question.lower() else None,
    )
    result = demos.run_demo("10")
    assert result.status == "pass"
    assert (
        "Unique file-backed test instruction."
        in fake_models[0]["messages"][0]["content"]
    )
    assert result.trace[0]["sha256"] == loaded["sha256"]
    assert loaded["sha256"] != load_skill()["sha256"]


def test_skill_loader_rejects_missing_or_invalid_files_and_nonmatching_requests(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        load_skill(tmp_path / "SKILL.md")
    path = tmp_path / "SKILL.md"
    path.write_text("No frontmatter")
    with pytest.raises(ValueError, match="frontmatter"):
        load_skill(path)
    assert skill_for("What is the notebook price?") is None
    assert skill_for("Do not cancel my order") is None
    assert skill_for("Keep my order") is None


def test_native_tool_response_without_call_fails_without_json_fallback(
    fake_models, monkeypatch
):
    monkeypatch.setattr(
        ollama,
        "chat",
        lambda *args, **kwargs: ollama.ToolCompletion(
            output='{"tool": "get_product", "product_id": "SKU-NOTEBOOK"}',
            tool_calls=[],
            model="test",
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=2,
            latency_ms=1,
            raw_response={
                "message": {"role": "assistant", "content": "An ordinary JSON answer"}
            },
        ),
    )
    result = demos.run_demo("08")
    assert result.status == "fail"
    assert result.usage["tool_calls"] == 0
    assert not result.checks[0]["selection_correct"]


def test_native_wrong_arguments_are_not_executed(fake_models, monkeypatch):
    monkeypatch.setattr(
        ollama,
        "chat",
        lambda *args, **kwargs: ollama.ToolCompletion(
            output="",
            tool_calls=[],
            model="test",
            finish_reason="stop",
            prompt_tokens=2,
            completion_tokens=2,
            latency_ms=1,
            raw_response={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_product",
                                "arguments": {"product_id": "SKU-BACKPACK"},
                            }
                        }
                    ],
                }
            },
        ),
    )
    result = demos.run_demo("08")
    assert result.status == "fail" and result.usage["tool_calls"] == 0


def test_missing_skill_is_execution_error_without_inference(fake_models, monkeypatch):
    def missing(question):
        raise FileNotFoundError("Missing SKILL.md")

    monkeypatch.setattr(demos, "skill_for", missing)
    result = demos.run_demo("10")
    assert result.status == "execution_error"
    assert "SKILL.md" in result.output["error"] and not fake_models


def test_skill_metadata_must_match_folder(tmp_path):
    path = tmp_path / "SKILL.md"
    path.write_text(
        "---\nname: wrong-name\ndescription: Cancel an order\n---\nInstructions\n"
    )
    with pytest.raises(ValueError, match="metadata"):
        load_skill(path)
