"""Minimal OpenRouter chat-completions client using only the standard library."""

import json
import os
import re
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ai_quality_evals.providers.ollama import Completion, ToolCompletion

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ENV_PATH = Path(__file__).resolve().parents[4] / ".env"
ENV_KEYS = {
    "AI_EVAL_PROVIDER",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_SECONDARY_MODEL",
    "OPENROUTER_JUDGE_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
}


def load_dotenv(path: Path = ENV_PATH) -> None:
    """Load supported local settings without replacing existing environment values."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        key = key.strip()
        if (
            not separator
            or key not in ENV_KEYS
            or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
        ):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _messages_for_api(messages: list[dict]) -> list[dict]:
    """Adapt the lab's provider-neutral tool-result shape to chat completions."""
    adapted = []
    last_tool_id = None
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls"):
            # Chat Completions expects the assistant's tool request to have
            # null content when it contains only tool calls.
            if not item.get("content"):
                item["content"] = None
            normalized_calls = []
            for call in item["tool_calls"]:
                function = call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    function = {**function, "arguments": json.dumps(arguments)}
                call_id = call.get("id", "call_unknown")
                last_tool_id = call_id
                normalized_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": function,
                    }
                )
            item["tool_calls"] = normalized_calls
        elif item.get("role") == "tool":
            item.pop("tool_name", None)
            item["tool_call_id"] = (
                item.get("tool_call_id") or last_tool_id or "call_unknown"
            )
        adapted.append(item)
    return adapted


def chat(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 128,
    timeout: float = 90,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
) -> Completion | ToolCompletion:
    load_dotenv()
    if max_tokens <= 0 or timeout <= 0:
        raise ValueError("max_tokens and timeout must be positive")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OpenRouter selected but OPENROUTER_API_KEY is not set")

    payload = {
        "model": model,
        "messages": _messages_for_api(messages),
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "demoshop_result",
                "strict": True,
                "schema": response_format,
            },
        }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if response_format is not None or tools is not None:
        payload["provider"] = {"require_parameters": True}

    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "DemoShop Evaluation Lab",
        },
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = json.load(response)
    except HTTPError as exc:
        # Keep the provider's short validation message in the saved trace so
        # request-shape failures can be diagnosed without exposing headers.
        try:
            body = json.loads(exc.read().decode("utf-8"))
            detail = body.get("error", {}).get("message")
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            detail = None
        suffix = f": {detail[:500]}" if isinstance(detail, str) and detail else ""
        raise RuntimeError(f"OpenRouter HTTP error {exc.code}{suffix}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"OpenRouter request failed ({type(exc).__name__})") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("choices"), list):
        raise RuntimeError("OpenRouter returned an invalid completion")
    if not raw["choices"] or not isinstance(raw["choices"][0], dict):
        raise RuntimeError("OpenRouter returned no completion choice")
    choice = raw["choices"][0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenRouter completion has no message")
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise RuntimeError("OpenRouter returned non-text content")

    normalized_calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise RuntimeError("OpenRouter returned malformed tool_calls")
        function = call["function"]
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise RuntimeError(
                    "OpenRouter returned malformed tool arguments"
                ) from exc
        if not isinstance(arguments, dict):
            raise RuntimeError("OpenRouter tool arguments must be a JSON object")
        normalized_calls.append(
            {
                "id": call.get("id", "call_unknown"),
                "type": "function",
                "function": {"name": function.get("name"), "arguments": arguments},
            }
        )
    usage = raw.get("usage") or {}
    common = {
        "output": content,
        "model": raw.get("model", model),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
        "latency_ms": (perf_counter() - started) * 1000,
        "raw_response": raw,
    }
    if tools is not None:
        return ToolCompletion(**common, tool_calls=normalized_calls)
    if not content.strip():
        raise RuntimeError("OpenRouter returned an empty answer")
    return Completion(**common)
