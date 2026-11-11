"""Transactional, isolated shop state; no model or framework grants permissions."""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from ai_quality_evals_advanced.contracts import FIXTURE
from ai_quality_evals_advanced.security import redact


def snapshot() -> dict:
    return json.loads(FIXTURE.read_bytes())["shop_snapshot"]


class ShopStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, case_id TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS orders (
                run_id TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (run_id, id));
            CREATE TABLE IF NOT EXISTS audit (
                run_id TEXT NOT NULL, actor TEXT NOT NULL, tool TEXT NOT NULL,
                outcome TEXT NOT NULL, changed INTEGER NOT NULL);
        """)

    def close(self):
        self.db.close()

    def start(self, run_id: str, case_id: str):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "INSERT INTO runs VALUES (?, ?, 'running')", (run_id, case_id)
            )
            self.db.executemany(
                "INSERT INTO orders VALUES (?, ?, ?)",
                [
                    (run_id, order["order_id"], json.dumps(order))
                    for order in snapshot()["orders"]
                ],
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def finish(self, run_id: str, status: str):
        self.db.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def session(self, run_id: str, actor: str = "C001"):
        return ShopSession(self, run_id, actor)


class ShopSession:
    def __init__(self, store: ShopStore, run_id: str, actor: str):
        self.store, self.run_id, self.actor = store, run_id, actor

    def get_product(self, product_id: str) -> dict:
        for product in snapshot()["products"]:
            if product["product_id"] == product_id:
                return product
        raise ValueError("Unknown product")

    def _order(self, order_id: str) -> dict:
        row = self.store.db.execute(
            "SELECT payload FROM orders WHERE run_id=? AND id=?",
            (self.run_id, order_id),
        ).fetchone()
        if row is None:
            raise ValueError("Order unavailable")
        order = json.loads(row[0])
        if order["customer_id"] != self.actor:
            raise PermissionError("Order access denied")
        return order

    def _audit(self, tool: str, outcome: str, changed: bool = False):
        self.store.db.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
            (self.run_id, self.actor, tool, outcome, int(changed)),
        )

    def get_order(self, order_id: str) -> dict:
        try:
            order = self._order(order_id)
        except (PermissionError, ValueError):
            self._audit("get_order", "denied")
            raise
        self._audit("get_order", "allowed")
        return order

    @staticmethod
    def _eligible(order: dict) -> bool:
        if order["status"] != "delivered" or not order.get("delivered_at_utc"):
            return False
        clock = datetime.fromisoformat(snapshot()["as_of_utc"])
        delivered = datetime.fromisoformat(order["delivered_at_utc"])
        if delivered.tzinfo is None:
            raise ValueError("Delivery timestamp requires an explicit timezone")
        return timedelta(0) <= clock - delivered <= timedelta(days=14)

    def check_return_eligibility(self, order_id: str) -> dict:
        order = self.get_order(order_id)
        return {"order_id": order_id, "eligible": self._eligible(order)}

    def _change(self, order_id: str, returning: bool) -> dict:
        tool = "request_return" if returning else "cancel_order"
        target = "return requested" if returning else "cancelled"
        db = self.store.db
        db.execute("BEGIN IMMEDIATE")
        try:
            order = self._order(order_id)
            changed = order["status"] != target
            if changed:
                allowed = (
                    self._eligible(order)
                    if returning
                    else order["status"] == "awaiting shipment"
                )
                if not allowed:
                    raise PermissionError("Action denied by shop policy")
                order["status"] = target
                if returning:
                    order["return_id"] = uuid.uuid4().hex
                db.execute(
                    "UPDATE orders SET payload=? WHERE run_id=? AND id=?",
                    (json.dumps(order), self.run_id, order_id),
                )
            self._audit(tool, "allowed", changed)
            db.execute("COMMIT")
            return {
                "order_id": order_id,
                "status": target,
                "changed": changed,
                **({"return_id": order["return_id"]} if returning else {}),
            }
        except BaseException:
            db.execute("ROLLBACK")
            self._audit(tool, "denied")
            raise

    def cancel_order(self, order_id: str) -> dict:
        return self._change(order_id, False)

    def request_return(self, order_id: str) -> dict:
        return self._change(order_id, True)


class ToolGateway:
    def __init__(
        self, session: ShopSession, allowed: set[str], limit: int = 5, telemetry=None
    ):
        self.session, self.allowed, self.limit = session, frozenset(allowed), limit
        self.telemetry = telemetry
        self.calls = 0

    def call(self, name: str, arguments: dict) -> dict:
        if self.telemetry is None:
            return self._dispatch(name, arguments)
        safe_name = (
            name
            if name
            in {
                "get_product",
                "get_order",
                "check_return_eligibility",
                "cancel_order",
                "request_return",
            }
            else "invalid"
        )
        with self.telemetry.span(
            "tool." + safe_name, "TOOL", **{"tool.name": safe_name}
        ) as span:
            capture = getattr(self.telemetry, "capture_content", False)
            if capture:
                safe_arguments = redact(arguments)
                span.set_attribute("input.value", json.dumps(safe_arguments))
                self.telemetry.log(
                    "tool.call.started", tool=safe_name, arguments=safe_arguments
                )
            try:
                result = self._dispatch(name, arguments)
                span.set_attribute("security.decision", "allowed")
                if capture:
                    safe_result = redact(result)
                    span.set_attribute("output.value", json.dumps(safe_result))
                    self.telemetry.log(
                        "tool.call.completed", tool=safe_name, result=safe_result
                    )
                if "changed" in result:
                    span.set_attribute("tool.state_changed", result["changed"])
                return result
            except Exception as exc:
                span.set_attribute("security.decision", "denied")
                if capture:
                    self.telemetry.log(
                        "tool.call.failed", tool=safe_name, error=type(exc).__name__
                    )
                raise

    def _dispatch(self, name: str, arguments: dict) -> dict:
        if self.calls >= self.limit:
            raise RuntimeError("Tool-call budget exhausted")
        self.calls += 1
        parameters = {
            "get_product": "product_id",
            "get_order": "order_id",
            "cancel_order": "order_id",
            "request_return": "order_id",
            "check_return_eligibility": "order_id",
        }
        if name not in parameters or name not in self.allowed:
            self.session._audit("gateway", "capability_denied")
            raise PermissionError("Tool is not allowed in this execution mode")
        parameter = parameters[name]
        if set(arguments) != {parameter} or not isinstance(arguments[parameter], str):
            self.session._audit(name, "arguments_denied")
            raise ValueError(
                "Invalid tool arguments; caller identity is host-controlled"
            )
        return getattr(self.session, name)(**arguments)
