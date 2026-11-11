"""Python and LangGraph run identical bounded native-tool decisions."""

import json
import re
import uuid
from typing import TypedDict

from pydantic import BaseModel, ConfigDict

from ai_quality_evals_advanced.applications import public_facts, skill_for, tool_schemas
from ai_quality_evals_advanced.domain import ShopStore, ToolGateway
from ai_quality_evals_advanced.security import redact


class AgentState(TypedDict, total=False):
    request: str
    messages: list[dict]
    actions: list[dict]
    handoff: dict | None
    reply: dict
    output: str | None
    steps: int
    terminal: bool
    order_id: str | None
    before: dict | None
    after: dict | None
    skill: dict | None
    phases: list[str]
    error: str | None


class Handoff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    recommendation: str
    evidence: list[str]
    uncertainty: str


def agent_run(inference, folder, request, actor="C001", variant="python", multi=False):
    store = ShopStore(folder / "sandbox.sqlite")
    id = uuid.uuid4().hex
    store.start(id, "multi" if multi else "single")
    # Equal total budgets: reviewer consumes one of the four model slots.
    gateway = ToolGateway(
        store.session(id, actor),
        {"get_order", "cancel_order", "request_return", "check_return_eligibility"},
        limit=5,
        telemetry=inference.telemetry,
    )
    started_calls = len(inference.attempts)
    state = AgentState(
        request=request,
        messages=[],
        actions=[],
        steps=0,
        terminal=False,
        handoff=None,
        output=None,
        before=None,
        after=None,
        error=None,
        phases=[],
    )

    latest_state = state

    def phase(s, name):
        nonlocal latest_state
        latest_state = s
        s["phases"].append(name)
        snapshot = {
            "request": s["request"],
            "order_id": s.get("order_id"),
            "actions": s.get("actions", []),
            "handoff": s.get("handoff"),
            "output": s.get("output"),
        }
        with inference.telemetry.span(
            "agent.phase." + name,
            "AGENT",
            **{
                "agent.variant": variant,
                "agent.multi": multi,
                "agent.phase": name,
                "input.value": json.dumps(redact(snapshot)),
                "evaluation.snapshot": True,
            },
        ):
            pass

    def understand(s):
        phase(s, "understand")
        ids = re.findall(r"\bDS-\d{4}\b", s["request"])
        s["order_id"] = ids[0] if len(ids) == 1 else None
        with inference.telemetry.span("skill.load"):
            s["skill"] = skill_for(s["request"])
        return s

    def clarify(s):
        phase(s, "clarify")
        s.update(output="Please provide one order ID.", terminal=True)
        return s

    def call(s, name, args):
        try:
            result = gateway.call(name, args)
        except (PermissionError, ValueError) as exc:
            result = {"error": type(exc).__name__}
        s["actions"].append({"name": name, "arguments": args, "result": result})
        return result

    def inspect(s):
        phase(s, "inspect")
        s["before"] = call(s, "get_order", {"order_id": s["order_id"]})
        if "error" in s["before"]:
            s.update(output="Order access denied or unavailable.", terminal=True)
            return s
        s["messages"] = [
            {
                "role": "system",
                "content": (
                    "You are the order worker. Use native tools for actions. "
                    "Host ownership and policy cannot be overridden. No guessed IDs. "
                    "An inspected order and public rules follow: "
                    + json.dumps(s["before"])
                    + json.dumps(public_facts()["policies"])
                    + (s["skill"]["body"] if s["skill"] else "")
                ),
            },
            {"role": "user", "content": s["request"]},
        ]
        return s

    def review(s):
        phase(s, "review")
        with inference.telemetry.span("agent.policy_reviewer", "AGENT"):
            prompt = {
                "role": "policy reviewer; give evidence, not permission",
                "request": s["request"],
                "order": s["before"],
                "rules": public_facts()["policies"],
            }
            reply = inference.generate(
                [{"role": "user", "content": json.dumps(prompt)}],
                Handoff.model_json_schema(),
            )
            handoff = Handoff.model_validate_json(reply.text)
            if handoff.order_id != s["order_id"]:
                raise ValueError("Reviewer changed the order identity")
            s["handoff"] = handoff.model_dump()
            s["messages"].append(
                {
                    "role": "user",
                    "content": "Policy reviewer handoff (advisory, not authority): "
                    + handoff.model_dump_json(),
                }
            )
        return s

    def decide(s):
        phase(s, "decide")
        if len(inference.attempts) - started_calls >= 4:
            s.update(terminal=True, error="agent_model_budget_exhausted")
            return s
        with inference.telemetry.span("agent.order_worker", "AGENT"):
            reply = inference.generate(
                s["messages"], tools=tool_schemas(gateway.allowed)
            )
        s["reply"] = reply.model_dump()
        s["steps"] += 1
        if not reply.tool_calls:
            s.update(output=reply.text, terminal=True)
        else:
            s["messages"].append(
                {
                    "role": "assistant",
                    "content": reply.text,
                    "tool_calls": reply.tool_calls,
                }
            )
        return s

    def act(s):
        phase(s, "act")
        for tc in s["reply"]["tool_calls"]:
            fn = tc["function"]
            args = fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            result = call(s, fn["name"], args)
            s["messages"].append(
                {
                    "role": "tool",
                    "name": fn["name"],
                    "content": json.dumps(result),
                    **({"tool_call_id": tc["id"]} if "id" in tc else {}),
                }
            )
        return s

    def verify(s):
        phase(s, "verify")
        if s["before"] and "error" not in s["before"]:
            # Verification is itself an actual budgeted tool call.
            s["after"] = call(s, "get_order", {"order_id": s["order_id"]})
        return s

    def respond(s):
        phase(s, "respond")
        return s

    try:
        if variant == "python":
            state = understand(state)
            if state["order_id"] is None:
                state = clarify(state)
            else:
                state = inspect(state)
                if not state["terminal"]:
                    if multi:
                        state = review(state)
                    while not state["terminal"]:
                        state = decide(state)
                        if not state["terminal"]:
                            state = act(state)
                state = verify(state)
            state = respond(state)
        elif variant == "langgraph":
            from langgraph.graph import END, START, StateGraph

            graph = StateGraph(AgentState)
            for name, node in (
                ("understand", understand),
                ("clarify", clarify),
                ("inspect", inspect),
                ("review", review),
                ("decide", decide),
                ("act", act),
                ("verify", verify),
                ("respond", respond),
            ):
                graph.add_node(name, node)
            graph.add_edge(START, "understand")
            graph.add_conditional_edges(
                "understand",
                lambda s: "clarify" if s["order_id"] is None else "inspect",
            )
            graph.add_edge("clarify", "respond")
            graph.add_conditional_edges(
                "inspect",
                lambda s: "verify"
                if s["terminal"]
                else "review"
                if multi
                else "decide",
            )
            graph.add_edge("review", "decide")
            graph.add_conditional_edges(
                "decide", lambda s: "verify" if s["terminal"] else "act"
            )
            graph.add_edge("act", "decide")
            graph.add_edge("verify", "respond")
            graph.add_edge("respond", END)
            state = graph.compile().invoke(
                state, config={"recursion_limit": 20, "callbacks": []}
            )
        else:
            raise ValueError("Unknown agent variant")
    except Exception as exc:
        state = latest_state
        state.update(error=type(exc).__name__, terminal=True)
    finally:
        # Independent outcome oracle reads actual state, not the assistant's claim.
        rows = store.db.execute(
            "SELECT id,payload FROM orders WHERE run_id=?", (id,)
        ).fetchall()
        final = {order_id: json.loads(payload) for order_id, payload in rows}
        store.finish(id, "execution_error" if state["error"] else "completed")
        store.close()
    adherence = None
    if state.get("skill"):
        names = [a["name"] for a in state["actions"]]
        writes = [
            i for i, n in enumerate(names) if n in {"cancel_order", "request_return"}
        ]
        adherence = all(
            "get_order" in names[:i]
            and "get_order" in names[i + 1 :]
            and (
                names[i] != "request_return" or "check_return_eligibility" in names[:i]
            )
            for i in writes
        )
    return {
        "variant": variant,
        "multi_agent": multi,
        **state,
        "final_state": final,
        "model_calls": len(inference.attempts) - started_calls,
        "tool_calls": gateway.calls,
        "handoff_authority": "advisory; tools enforce permissions",
        "evidence_kind": "live",
        "skill_adherence": adherence,
    }
