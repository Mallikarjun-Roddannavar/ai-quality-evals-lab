"""Owned synthetic promptfoo target: actually execute domain authorization."""

import json
import sys
import uuid

from ai_quality_evals_advanced.contracts import ARTIFACTS
from ai_quality_evals_advanced.domain import ShopStore, ToolGateway
from ai_quality_evals_advanced.security import redact


def target(payload, database):
    store = ShopStore(database)
    id = uuid.uuid4().hex
    store.start(id, "promptfoo-security")
    try:
        gateway = ToolGateway(
            store.session(id, "C001"), {"get_order", "cancel_order"}, 1
        )
        try:
            result = gateway.call(payload["tool"], {"order_id": payload["order_id"]})
            return {"status": "allowed", "changed": result.get("changed", False)}
        except (PermissionError, ValueError):
            return {"status": "denied", "changed": False}
    finally:
        store.finish(id, "completed")
        store.close()


if __name__ == "__main__":
    print(
        json.dumps(
            redact(target(json.loads(sys.stdin.read()), ARTIFACTS / "security.sqlite"))
        )
    )
