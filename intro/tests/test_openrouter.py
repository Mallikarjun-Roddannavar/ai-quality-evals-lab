"""OpenRouter adapter tests use mocked HTTP and never need credentials."""

import io
import json
import os
from urllib.error import HTTPError

import pytest

from ai_quality_evals.providers import openrouter


def test_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(openrouter, "load_dotenv", lambda: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("No request should be sent without an API key")

    monkeypatch.setattr(openrouter, "urlopen", unexpected)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        openrouter.chat("provider/model", [{"role": "user", "content": "hi"}])


def test_dotenv_loads_supported_values_without_overriding_shell(monkeypatch, tmp_path):
    settings = tmp_path / ".env"
    settings.write_text(
        "AI_EVAL_PROVIDER=openrouter\n"
        "OPENROUTER_API_KEY=dotenv-key\n"
        "UNRELATED_VALUE=ignored\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-key")
    monkeypatch.delenv("AI_EVAL_PROVIDER", raising=False)

    openrouter.load_dotenv(settings)

    assert os.environ["AI_EVAL_PROVIDER"] == "openrouter"
    assert os.environ["OPENROUTER_API_KEY"] == "shell-key"
    assert "UNRELATED_VALUE" not in os.environ


def test_chat_request_records_usage_and_never_exposes_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    captured = {}

    def respond(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return io.BytesIO(
            json.dumps(
                {
                    "model": "provider/model:free",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "INR 100"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "cost": 0.0000465,
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(openrouter, "urlopen", respond)
    result = openrouter.chat(
        "provider/model",
        [{"role": "user", "content": "Price?"}],
        response_format={"type": "object", "properties": {}},
    )

    assert result.output == "INR 100"
    assert result.model == "provider/model:free"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 3
    assert result.cost == 0.0000465
    assert captured["url"] == openrouter.ENDPOINT
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
    assert captured["payload"]["provider"]["require_parameters"] is True
    assert captured["payload"]["seed"] == 42
    assert captured["timeout"] == 90
    assert "test-secret" not in result.model_dump_json()


def test_native_tool_calls_parse_arguments_and_tool_result_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    captured = {}

    def respond(request, timeout):
        captured["payload"] = json.loads(request.data)
        return io.BytesIO(
            json.dumps(
                {
                    "model": "provider/tool-model",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_product",
                                            "arguments": json.dumps(
                                                {"product_id": "SKU-NOTEBOOK"}
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 5},
                }
            ).encode()
        )

    monkeypatch.setattr(openrouter, "urlopen", respond)
    tools = [
        {
            "type": "function",
            "function": {"name": "get_product", "parameters": {"type": "object"}},
        }
    ]
    result = openrouter.chat(
        "provider/tool-model",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "get_product",
                            "arguments": {"product_id": "SKU-NOTEBOOK"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_name": "get_product",
                "content": '{"price_inr":100}',
            },
        ],
        tools=tools,
    )

    assert result.tool_calls[0]["function"]["arguments"] == {
        "product_id": "SKU-NOTEBOOK"
    }
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["provider"]["require_parameters"] is True
    sent_call = captured["payload"]["messages"][-2]["tool_calls"][0]
    assert json.loads(sent_call["function"]["arguments"]) == {
        "product_id": "SKU-NOTEBOOK"
    }
    assert captured["payload"]["messages"][0]["content"] is None
    assert captured["payload"]["messages"][-1]["tool_call_id"] == "call-1"
    assert "name" not in captured["payload"]["messages"][-1]
    assert "tool_name" not in captured["payload"]["messages"][-1]


def test_invalid_tool_arguments_are_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    def respond(request, timeout):
        return io.BytesIO(
            json.dumps(
                {
                    "model": "provider/tool-model",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_product",
                                            "arguments": "not-json",
                                        }
                                    }
                                ]
                            },
                        }
                    ],
                    "usage": {},
                }
            ).encode()
        )

    monkeypatch.setattr(openrouter, "urlopen", respond)
    with pytest.raises(RuntimeError, match="malformed tool arguments"):
        openrouter.chat("provider/tool-model", [], tools=[])


def test_http_error_includes_provider_message_without_headers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    def reject(request, timeout):
        raise HTTPError(
            openrouter.ENDPOINT,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"Invalid tool result message"}}'),
        )

    monkeypatch.setattr(openrouter, "urlopen", reject)
    with pytest.raises(
        RuntimeError, match="HTTP error 400: Invalid tool result message"
    ) as exc:
        openrouter.chat("provider/tool-model", [{"role": "user", "content": "x"}])
    assert "test-secret" not in str(exc.value)
