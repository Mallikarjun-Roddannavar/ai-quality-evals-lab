"""Shared DemoShop modes, bounded model calls and resettable local tools."""

import asyncio
import copy
import json
import re
import sys
from pathlib import Path
from typing import Literal

from ai_quality_evals.foundations import Contract, Nonblank, sha256
from ai_quality_evals.providers import ollama, openrouter

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "intro/datasets/demoshop/foundations-v1.json"
SKILL = ROOT / "intro/skills/cancel-owned-order/SKILL.md"


def snapshot() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["shop_snapshot"]


class Action(Contract):
    tool: Literal["get_product", "get_order", "cancel_order", "finish"]
    product_id: str | None = None
    order_id: str | None = None
    answer: str | None = None


class PolicyReview(Contract):
    verdict: Literal["allowed", "not_allowed", "uncertain"]
    reason: Nonblank


class Shop:
    """Trusted caller context is separate from all model-provided arguments."""

    def __init__(self, caller: str = "C001"):
        self.data = copy.deepcopy(snapshot())
        self.caller = caller

    def get_product(self, product_id: str) -> dict:
        for product in self.data["products"]:
            if product["product_id"] == product_id:
                return copy.deepcopy(product)
        raise ValueError("Unknown product")

    def get_order(self, order_id: str) -> dict:
        for order in self.data["orders"]:
            if order["order_id"] == order_id:
                if order["customer_id"] != self.caller:
                    raise PermissionError(
                        "Access denied: order does not belong to caller"
                    )
                return copy.deepcopy(order)
        raise ValueError("Unknown order")

    def cancel_order(self, order_id: str) -> dict:
        order = self.get_order(order_id)
        if order["status"] == "cancelled":
            return {"order_id": order_id, "status": "cancelled", "changed": False}
        if order["status"] != "awaiting shipment":
            raise PermissionError("Cancellation denied: order has already shipped")
        for stored in self.data["orders"]:
            if stored["order_id"] == order_id:
                stored["status"] = "cancelled"
        return {"order_id": order_id, "status": "cancelled", "changed": True}


class Runtime:
    def __init__(
        self,
        model: str = "gemma4:e2b",
        *,
        max_calls: int = 4,
        max_tools: int = 3,
        caller: str = "C001",
        provider: str = "ollama",
    ):
        self.model = model
        if provider not in {"ollama", "openrouter"}:
            raise ValueError("provider must be 'ollama' or 'openrouter'")
        self.provider = provider
        self.max_calls = max_calls
        self.max_tools = max_tools
        self.shop = Shop(caller)
        self.events: list[dict] = []
        self.models: dict[str, dict] = {}

    def infer(
        self,
        messages: list[dict],
        *,
        role: str = "Assistant",
        model: str | None = None,
        schema: dict | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        if sum(e["kind"] == "model" for e in self.events) >= self.max_calls:
            raise RuntimeError("Model-call budget exhausted")
        selected = model or self.model
        event = {
            "kind": "model",
            "role": role,
            "model": selected,
            "messages": copy.deepcopy(messages),
            "completion": None,
        }
        self.events.append(event)
        try:
            if selected not in self.models:
                self.models[selected] = (
                    ollama.installed_model("http://localhost:11434", selected)
                    if self.provider == "ollama"
                    else {"provider": "openrouter", "model": selected}
                )
            event["generation_settings"] = {
                "format": schema,
                "tools": tools,
                "think": False
                if "thinking" in self.models[selected].get("capabilities", [])
                else None,
            }
            if self.provider == "ollama":
                completion = ollama.chat(
                    "http://localhost:11434",
                    selected,
                    messages,
                    max_tokens=256,
                    timeout=90,
                    response_format=schema,
                    tools=tools,
                    thinking=False
                    if "thinking" in self.models[selected].get("capabilities", [])
                    else None,
                )
            else:
                completion = openrouter.chat(
                    selected,
                    messages,
                    max_tokens=256,
                    timeout=90,
                    response_format=schema,
                    tools=tools,
                )
            event["completion"] = completion.model_dump()
            completed_tool_request = (
                tools is not None
                and completion.finish_reason in {"tool_calls", "stop"}
                and bool(completion.tool_calls)
            )
            if completion.finish_reason != "stop" and not completed_tool_request:
                raise RuntimeError(
                    "Model output may be truncated; try a smaller example"
                )
            if tools is not None:
                if self.provider == "openrouter":
                    return json.dumps(
                        {
                            "role": "assistant",
                            "content": completion.output,
                            "tool_calls": completion.tool_calls,
                        }
                    )
                return json.dumps(completion.raw_response["message"])
            return completion.output
        except Exception as exc:
            event["error"] = str(exc)
            raise

    def ask(
        self,
        question: str,
        *,
        context: dict | str | None = None,
        history: list[dict] | None = None,
        role: str = "Assistant",
    ) -> str:
        facts = (
            context
            if context is not None
            else {k: self.shop.data[k] for k in ("currency", "products", "policies")}
        )
        instruction = (
            "You are DemoShop's answer-only assistant. Use only the supplied facts. "
            "Answer briefly. Unknown information must remain unknown. Do not invent "
            "refund terms or claim tools/actions you did not perform. Facts:\n"
            + json.dumps(facts)
        )
        return self.infer(
            [
                {"role": "system", "content": instruction},
                *(history or []),
                {"role": "user", "content": question},
            ],
            role=role,
        )

    def tool(self, action: Action) -> dict:
        if sum(e["kind"] == "tool" for e in self.events) >= self.max_tools:
            raise RuntimeError("Tool-call budget exhausted")
        if action.tool == "finish":
            raise ValueError("finish is not an executable shop tool")
        event = {
            "kind": "tool",
            "tool": action.tool,
            "action": action.model_dump(),
            "caller": self.shop.caller,
        }
        self.events.append(event)
        try:
            if action.tool == "get_product":
                result = self.shop.get_product(action.product_id or "")
            elif action.tool == "get_order":
                result = self.shop.get_order(action.order_id or "")
            else:
                result = self.shop.cancel_order(action.order_id or "")
        except (ValueError, PermissionError) as exc:
            result = {"error": str(exc)}
        event["result"] = result
        return result

    def usage(self) -> dict:
        model_events = [e for e in self.events if e["kind"] == "model"]
        completed = [e["completion"] for e in model_events if e["completion"]]
        costs = [c.get("cost") for c in completed]
        provider_cost_known = (
            bool(model_events)
            and len(completed) == len(model_events)
            and all(cost is not None for cost in costs)
        )
        reported_cost = (
            sum(costs)
            if self.provider == "openrouter" and provider_cost_known
            else None
        )
        return {
            "model_calls": len(model_events),
            "tool_calls": sum(e["kind"] == "tool" for e in self.events),
            "latency_ms": sum(c["latency_ms"] for c in completed),
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in completed),
            "completion_tokens": sum(c["completion_tokens"] or 0 for c in completed),
            "missing_token_measurements": len(completed) != len(model_events)
            or any(
                c["prompt_tokens"] is None or c["completion_tokens"] is None
                for c in completed
            ),
            "cost": reported_cost,
            "cost_currency": "USD" if reported_cost is not None else None,
            "missing_cost_measurements": self.provider == "openrouter"
            and bool(model_events)
            and not provider_cost_known,
            "cost_note": (
                "Sum of OpenRouter-reported USD charges; local energy cost unmeasured"
                if reported_cost is not None
                else "OpenRouter cost missing for one or more calls; unknown, not zero"
                if self.provider == "openrouter"
                else "Local hardware and energy cost unmeasured"
            ),
        }


def retrieve(question: str, *, wrong_document: bool = False) -> dict:
    data = snapshot()
    documents = {
        "catalog": "Product prices and stock: " + json.dumps(data["products"]),
        "returns": data["policies"]["return_rule"],
        "cancellation": data["policies"]["cancellation_rule"],
    }

    def words(text: str) -> set[str]:
        text = (
            text.lower().replace("cancellation", "cancel").replace("returns", "return")
        )
        return set(re.findall(r"[a-z0-9]+", text)) - {
            "the",
            "a",
            "is",
            "can",
            "i",
            "when",
            "of",
            "and",
            "to",
            "an",
            "my",
        }

    scores = {
        key: len(words(question) & words(body)) for key, body in documents.items()
    }
    selected = "catalog" if wrong_document else max(scores, key=scores.get)
    return {"document_id": selected, "text": documents[selected], "scores": scores}


def load_skill(path: Path = SKILL) -> dict:
    """Load the bundled skill's simple name/description YAML header and Markdown.

    Only this two-field scalar header is supported; this is not a general YAML
    parser or a runtime for arbitrary third-party skill scripts.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(
            "SKILL.md must have YAML frontmatter and Markdown instructions"
        )
    header, body = text[4:].split("\n---\n", 1)
    metadata = {}
    for line in header.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in {"name", "description"} or key in metadata:
            raise ValueError("Bundled skill supports only name and description scalars")
        metadata[key] = value.strip()
    if (
        metadata.get("name") != path.parent.name
        or len(metadata.get("name", "")) > 64
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", metadata.get("name", ""))
        or not metadata.get("description")
        or len(metadata.get("description", "")) > 1024
        or not body.strip()
    ):
        raise ValueError("Invalid bundled skill metadata or empty instructions")
    version = re.search(r"^Version: (\d+\.\d+\.\d+)$", body, re.MULTILINE)
    if version is None:
        raise ValueError("Bundled skill must declare its Version in the Markdown body")
    return {
        **metadata,
        "version": version[1],
        "path": path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else str(path),
        "sha256": sha256(raw),
        "body": body.strip(),
        "instructions": re.findall(r"^\d+\. (.+)$", body, re.MULTILINE),
    }


def skill_for(question: str) -> dict | None:
    # Deliberately small host routing; instructions themselves come from disk.
    request = question.lower()
    if re.search(r"\b(don't|do not|never)\s+cancel\b|\bkeep\b", request):
        return None
    if re.search(r"\bcancel\b", request):
        return load_skill()
    return None


def run_agent(
    runtime: Runtime, *, multi: bool = False, skill: dict | None = None
) -> dict:
    task = "Cancel my order DS-1001 if the shop rules permit it."
    handoff = ""
    if multi:
        order = runtime.tool(Action(tool="get_order", order_id="DS-1001"))
        review_text = runtime.infer(
            [
                {
                    "role": "system",
                    "content": "You are the policy reviewer. Assess cancellation "
                    "using only the supplied order and policy. Return the "
                    "required JSON.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "order": order,
                            "policy": runtime.shop.data["policies"][
                                "cancellation_rule"
                            ],
                        }
                    ),
                },
            ],
            role="Policy reviewer",
            schema=PolicyReview.model_json_schema(),
        )
        review = PolicyReview.model_validate_json(review_text)
        handoff = review.model_dump_json()
        runtime.events.append(
            {
                "kind": "handoff",
                "from": "Policy reviewer",
                "to": "Order worker",
                "content": review.model_dump(),
            }
        )
    instructions = (
        "You are a bounded DemoShop order worker. Complete the user's task using "
        "JSON actions. Valid tools: get_order, cancel_order, finish. Caller identity "
        "is enforced by tools. First inspect the order unless an inspection is already "
        "provided. If an owned order is awaiting shipment, call cancel_order. "
        "Never claim cancellation without a successful tool result. "
        "Cancellation is allowed only before shipment. "
        + (
            "Inspection and reviewer handoff: " + json.dumps(order) + handoff
            if multi
            else ""
        )
    )
    if skill is not None:
        runtime.events.append(
            {"kind": "skill_load", "path": skill["path"], "sha256": skill["sha256"]}
        )
        instructions = (
            "Complete the request using this loaded SKILL.md. Return JSON actions "
            "for get_order, cancel_order or finish. Tool permissions are enforced "
            "by the host.\n\n" + skill["body"]
        )
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": task},
    ]
    last_answer = "No successful action yet."
    while sum(e["kind"] == "model" for e in runtime.events) < runtime.max_calls:
        inspected = any(
            e["kind"] == "tool"
            and e["tool"] == "get_order"
            and e.get("result", {}).get("order_id") == "DS-1001"
            for e in runtime.events
        )
        schema = Action.model_json_schema()
        schema["properties"]["tool"]["enum"] = (
            ["cancel_order", "finish"] if inspected else ["get_order"]
        )
        # OpenAI-compatible strict structured outputs require every declared
        # object property, including nullable fields, to appear in `required`.
        schema["required"] = list(schema["properties"])
        for property_schema in schema["properties"].values():
            property_schema.pop("default", None)
        answer = runtime.infer(
            messages,
            role="Order worker" if multi else "Single agent",
            schema=schema,
        )
        action = Action.model_validate_json(answer)
        if action.tool == "finish":
            last_answer = action.answer or "Agent finished."
            break
        result = runtime.tool(action)
        last_answer = json.dumps(result)
        messages.extend(
            [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Actual tool result: " + last_answer},
            ]
        )
        if runtime.shop.get_order("DS-1001")["status"] == "cancelled":
            break
    return {
        "task": task,
        "output": last_answer,
        "final_order": runtime.shop.get_order("DS-1001"),
        "handoff": handoff or None,
        "usage": runtime.usage(),
    }


async def mcp_lookup(product_id: str = "SKU-NOTEBOOK") -> dict:
    from mcp import Client, StdioServerParameters

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ai_quality_evals.mcp_server"],
        cwd=str(ROOT),
    )
    async with Client(parameters, read_timeout_seconds=15) as client:
        tools = await client.list_tools()
        good = await client.call_tool("get_product", {"product_id": product_id})
        bad = await client.call_tool("get_product", {"wrong_argument": product_id})
        return {
            "transport": "stdio",
            "tools": [t.name for t in tools.tools],
            "response": good.model_dump(mode="json"),
            "invalid_arguments": bad.model_dump(mode="json"),
        }


def serve_mcp() -> None:
    from ai_quality_evals.mcp_server import server

    server.run(transport="stdio")


if __name__ == "__main__":
    if "--mcp-server" in sys.argv:
        serve_mcp()
    else:
        print(json.dumps(asyncio.run(mcp_lookup()), indent=2))
