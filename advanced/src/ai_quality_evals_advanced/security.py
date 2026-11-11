"""Redaction is applied before artifacts, logs, telemetry and console output."""

import re

SENSITIVE = re.compile(
    r"authorization|api[_-]?key|password|secret|access[_-]?token", re.I
)
TOKENS = re.compile(r"\bsk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{8,}\b|Bearer\s+\S+", re.I)


def redact(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SENSITIVE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return TOKENS.sub("[REDACTED]", value)
    return value
