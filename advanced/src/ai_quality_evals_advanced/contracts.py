"""A1 contracts: one concrete price example and explicit runtime configuration."""

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "reports/advanced"
FIXTURE = ROOT / "intro/datasets/demoshop/foundations-v1.json"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class Settings(Contract):
    provider: Literal["ollama", "openrouter"] = "ollama"
    model: str = Field(default="gemma4:e2b", min_length=1)
    ollama_url: str = "http://127.0.0.1:11434"
    phoenix_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    export_traces: bool = True
    deadline_seconds: float = Field(default=120.0, gt=0, le=600)
    request_timeout: float = Field(default=30.0, gt=0, le=120)
    max_model_calls: int = Field(default=8, ge=1, le=100)
    max_tool_calls: int = Field(default=5, ge=1, le=50)
    max_tokens: int = Field(default=256, ge=16, le=4096)
    max_attempts: int = Field(default=2, ge=1, le=3)
    concurrency: int = Field(default=1, ge=1, le=4)
    retrieval: Literal["lexical", "semantic"] = "lexical"
    embedding_model: str | None = None

    @field_validator("ollama_url", "phoenix_endpoint")
    @classmethod
    def loopback_only(cls, value: str) -> str:
        parts = urlsplit(value)
        if (
            parts.scheme != "http"
            or parts.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("Local services require credential-free loopback HTTP")
        return value


class PriceCase(Contract):
    case_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    question: str = Field(min_length=1, max_length=2000)
    product_id: str
    controlled_answer: str


class Reply(Contract):
    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    tool_calls: list[dict] = Field(default_factory=list)


class RunResult(Contract):
    schema_version: str = "1.0.0"
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    case_id: str
    mode: Literal["offline", "live"]
    evidence_kind: Literal["authored_control", "live"]
    created_at_utc: str
    execution_status: Literal["completed", "execution_error", "cancelled"]
    verdict: Literal["pass", "fail", "needs_review", "execution_error"]
    output: str | None
    explanation: str
    source_facts: dict
    checks: list[dict]
    usage: dict
    attempts: list[dict]
    trace_id: str
    telemetry: dict
    provenance: dict


def run_folder(run_id: str, base: Path = ARTIFACTS) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Expected a run ID, not a filesystem path")
    path = base / run_id
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Run path escapes artifact storage")
    return path
