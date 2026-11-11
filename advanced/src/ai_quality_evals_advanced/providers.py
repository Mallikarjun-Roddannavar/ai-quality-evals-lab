"""Allowlisted HTTP inference with no redirects, SDK defaults or hidden retries."""

import http.client
import json
import os
import time
from urllib.parse import urlsplit

from ai_quality_evals_advanced.contracts import Reply, Settings
from ai_quality_evals_advanced.security import redact


class ProviderError(RuntimeError):
    def __init__(self, code: str, retryable: bool = False):
        super().__init__(code)
        self.retryable = retryable


def request(
    url: str, payload: dict | None, timeout: float, key: str | None = None
) -> dict:
    parts = urlsplit(url)
    local = parts.scheme == "http" and parts.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    router = (
        parts.scheme == "https"
        and parts.netloc == "openrouter.ai"
        and parts.path == "/api/v1/chat/completions"
    )
    if (
        not (local or router)
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ProviderError("destination_not_allowed")
    cls = http.client.HTTPSConnection if router else http.client.HTTPConnection
    conn = cls(parts.hostname, parts.port, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if key is not None:
        if not router:
            raise ProviderError("credentials_not_allowed_for_local_endpoint")
        headers["Authorization"] = "Bearer " + key
    try:
        conn.request(
            "GET" if payload is None else "POST",
            parts.path or "/",
            body=None if payload is None else json.dumps(payload),
            headers=headers,
        )
        response = conn.getresponse()
        if response.status != 200:
            raise ProviderError(
                f"http_{response.status}", response.status in {429, 500, 502, 503, 504}
            )
        content = response.read(4_000_001)
        if len(content) > 4_000_000:
            raise ProviderError("provider_response_too_large")
        data = json.loads(content)
        if not isinstance(data, dict) or "error" in data:
            raise ProviderError("invalid_provider_response")
        return data
    except ProviderError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise ProviderError("transport_error", True) from exc
    except (ValueError, UnicodeError) as exc:
        raise ProviderError("invalid_provider_json") from exc
    finally:
        conn.close()


class Inference:
    def __init__(self, config: Settings, telemetry):
        self.config, self.telemetry = config, telemetry
        self.deadline = time.monotonic() + config.deadline_seconds
        self.attempts: list[dict] = []
        self.metadata: dict = {}

    def remaining(self) -> float:
        value = self.deadline - time.monotonic()
        if value <= 0:
            raise ProviderError("deadline_exhausted")
        return min(value, self.config.request_timeout)

    def ready(self):
        c = self.config
        if c.provider == "openrouter":
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise ProviderError("openrouter_key_not_configured")
            self.metadata = {"model": c.model, "provider": "openrouter"}
            return
        base = c.ollama_url.rstrip("/")
        tags = request(base + "/api/tags", None, self.remaining())
        found = next(
            (m for m in tags.get("models", []) if m.get("name") == c.model), None
        )
        if found is None:
            raise ProviderError("model_not_installed_no_download_attempted")
        details = request(base + "/api/show", {"model": c.model}, self.remaining())
        version = request(base + "/api/version", None, self.remaining())
        self.metadata = {
            "model": c.model,
            "digest": found.get("digest"),
            "capabilities": details.get("capabilities", []),
            "runtime_version": version.get("version"),
        }

    def generate(
        self,
        messages: list[dict],
        schema: dict | None = None,
        tools: list[dict] | None = None,
    ) -> Reply:
        from opentelemetry.trace import StatusCode

        c = self.config
        if c.provider == "openrouter":
            from ai_quality_evals.providers.openrouter import _messages_for_api

            outbound_messages = _messages_for_api(messages)
        else:
            outbound_messages = messages
        for attempt in range(c.max_attempts):
            if len(self.attempts) >= c.max_model_calls:
                raise ProviderError("model_call_budget_exhausted")
            timeout = self.remaining()
            row = {"attempt": attempt + 1, "elapsed_ms": 0.0, "status": "started"}
            self.attempts.append(row)
            started = time.perf_counter()
            with self.telemetry.span(
                "model.generate",
                "LLM",
                **{
                    "gen_ai.provider.name": c.provider,
                    "gen_ai.request.model": c.model,
                    "attempt": attempt + 1,
                },
            ) as span:
                capture = getattr(self.telemetry, "capture_content", False)
                if capture:
                    safe_messages = redact(outbound_messages)
                    span.set_attribute(
                        "input.value", json.dumps(safe_messages, ensure_ascii=False)
                    )
                    span.set_attribute("input.mime_type", "application/json")
                    self.telemetry.log(
                        "model.call.started",
                        provider=c.provider,
                        model=c.model,
                        attempt=attempt + 1,
                        messages=safe_messages,
                        tools=redact(tools),
                        schema=redact(schema),
                    )
                try:
                    self.ready()
                    if c.provider == "ollama":
                        body = {
                            "model": c.model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": 0,
                                "seed": 42,
                                "num_predict": c.max_tokens,
                                "num_ctx": 2048,
                            },
                        }
                        if "thinking" in self.metadata.get("capabilities", []):
                            body["think"] = False
                        if schema is not None:
                            body["format"] = schema
                        if tools is not None:
                            body["tools"] = tools
                        raw = request(
                            c.ollama_url.rstrip("/") + "/api/chat", body, timeout
                        )
                        if raw.get("done") is not True:
                            raise ProviderError("incomplete_provider_response")
                        reply = Reply(
                            text=raw["message"]["content"],
                            model=raw["model"],
                            finish_reason=raw.get("done_reason"),
                            prompt_tokens=raw.get("prompt_eval_count"),
                            completion_tokens=raw.get("eval_count"),
                            tool_calls=raw["message"].get("tool_calls", []),
                        )
                    else:
                        raw = request(
                            "https://openrouter.ai/api/v1/chat/completions",
                            {
                                "model": c.model,
                                "messages": outbound_messages,
                                "temperature": 0,
                                "max_tokens": c.max_tokens,
                                **(
                                    {
                                        "response_format": {
                                            "type": "json_schema",
                                            "json_schema": {
                                                "name": "evaluation_response",
                                                "strict": True,
                                                "schema": schema,
                                            },
                                        },
                                    }
                                    if schema is not None
                                    else {}
                                ),
                                **({"tools": tools} if tools is not None else {}),
                                **(
                                    {"provider": {"require_parameters": True}}
                                    if schema is not None or tools is not None
                                    else {}
                                ),
                            },
                            timeout,
                            os.environ["OPENROUTER_API_KEY"],
                        )
                        choice = raw["choices"][0]
                        usage = raw.get("usage", {})
                        reply = Reply(
                            text=choice["message"].get("content") or "",
                            model=raw["model"],
                            finish_reason=choice.get("finish_reason"),
                            prompt_tokens=usage.get("prompt_tokens"),
                            completion_tokens=usage.get("completion_tokens"),
                            tool_calls=choice["message"].get("tool_calls", []),
                        )
                    if not reply.text.strip() and not reply.tool_calls:
                        raise ProviderError("empty_answer")
                    if capture:
                        safe_reply = redact(reply.model_dump())
                        span.set_attribute(
                            "output.value", json.dumps(safe_reply, ensure_ascii=False)
                        )
                        span.set_attribute("output.mime_type", "application/json")
                        self.telemetry.log(
                            "model.call.completed",
                            model=reply.model,
                            response=safe_reply,
                        )
                    self.metadata["last_response_model"] = reply.model
                    row.update(
                        status="completed",
                        prompt_tokens=reply.prompt_tokens,
                        completion_tokens=reply.completion_tokens,
                        finish_reason=reply.finish_reason,
                    )
                    for key, value in (
                        ("input", reply.prompt_tokens),
                        ("output", reply.completion_tokens),
                    ):
                        if value is not None:
                            span.set_attribute(f"gen_ai.usage.{key}_tokens", value)
                    return reply
                except ProviderError as exc:
                    if capture:
                        self.telemetry.log("model.call.failed", error=str(exc))
                    row.update(status="error", error=str(exc))
                    span.set_attribute("error.type", str(exc))
                    span.set_status(StatusCode.ERROR)
                    if not exc.retryable or attempt + 1 == c.max_attempts:
                        raise
                    time.sleep(min(0.1 * (2**attempt), self.remaining()))
                except (KeyError, TypeError, ValueError) as exc:
                    if capture:
                        self.telemetry.log(
                            "model.call.failed", error="invalid_response_schema"
                        )
                    row.update(status="error", error="invalid_response_schema")
                    raise ProviderError("invalid_response_schema") from exc
                finally:
                    row["elapsed_ms"] = (time.perf_counter() - started) * 1000
        raise ProviderError("attempts_exhausted")
