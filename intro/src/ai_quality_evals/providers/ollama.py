"""Small local Ollama HTTP client; no SDK, credentials or automatic downloads."""

import json
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import Field

from ai_quality_evals.foundations import Contract, Nonblank


class Completion(Contract):
    output: Nonblank
    model: Nonblank
    finish_reason: str | None
    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    cost: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    raw_response: dict


class ToolCompletion(Completion):
    """Native tool responses may have empty text and populated tool_calls."""

    output: str
    tool_calls: list[dict]


def local_url(base_url: str, endpoint: str) -> str:
    parts = urlsplit(base_url)
    if (
        parts.scheme != "http"
        or parts.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parts.username is not None
        or parts.password is not None
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
    ):
        raise ValueError("This pilot requires a local HTTP Ollama endpoint")
    return base_url.rstrip("/") + endpoint


def request_json(url: str, payload: dict | None, timeout: float) -> dict:
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error {exc.code}") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(f"Ollama request failed ({type(exc).__name__})") from exc
    if not isinstance(result, dict) or "error" in result:
        raise RuntimeError("Ollama returned an invalid/error response")
    return result


def installed_model(base_url: str, model: str) -> dict:
    data = request_json(local_url(base_url, "/api/tags"), None, 5)
    for entry in data.get("models", []):
        if entry.get("name") == model:
            runtime = request_json(local_url(base_url, "/api/version"), None, 5)
            details = request_json(
                local_url(base_url, "/api/show"), {"model": model}, 5
            )
            return {
                **entry,
                "ollama_version": runtime.get("version"),
                "capabilities": details.get("capabilities", []),
            }
    raise ValueError(f"Model {model!r} is not installed; no download was attempted")


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 128,
    timeout: float = 90,
    response_format: dict | None = None,
    thinking: bool | None = None,
    tools: list[dict] | None = None,
) -> Completion | ToolCompletion:
    if max_tokens <= 0 or timeout <= 0:
        raise ValueError("max_tokens and timeout must be positive")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": max_tokens,
            "num_ctx": 2048,
        },
    }
    if response_format is not None:
        payload["format"] = response_format
    if thinking is not None:
        payload["think"] = thinking
    if tools is not None:
        payload["tools"] = tools
    started = perf_counter()
    raw = request_json(local_url(base_url, "/api/chat"), payload, timeout)
    if raw.get("done") is not True:
        raise RuntimeError("Ollama response was not complete")
    message = raw.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("Ollama response has no text answer")
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise RuntimeError("Ollama returned malformed tool_calls")
    if not message["content"].strip() and not (tools is not None and calls):
        raise RuntimeError("Ollama returned an empty answer")
    cls = ToolCompletion if tools is not None else Completion
    extra = {"tool_calls": calls} if tools is not None else {}
    return cls(
        output=message["content"],
        model=raw["model"],
        finish_reason=raw.get("done_reason"),
        prompt_tokens=raw.get("prompt_eval_count"),
        completion_tokens=raw.get("eval_count"),
        latency_ms=(perf_counter() - started) * 1000,
        raw_response=raw,
        **extra,
    )
