"""Sixteen small demonstrations, common evidence and explicit replay."""

import asyncio
import json
import math
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import Field

from ai_quality_evals.evaluators.catalog import Product, check_catalog
from ai_quality_evals.evaluators.matching import exact_match, normalized_match
from ai_quality_evals.foundations import Contract, Nonblank, git_provenance, sha256
from ai_quality_evals.lab_systems import (
    FIXTURE,
    ROOT,
    SKILL,
    Action,
    Runtime,
    Shop,
    mcp_lookup,
    retrieve,
    run_agent,
    skill_for,
    snapshot,
)

REPORTS = ROOT / "reports/lab"
TOPICS = [
    ("01", "Foundations", "Compare an answer with an expected result.", "Chat"),
    ("02", "Deterministic", "Check a catalog fact without an LLM judge.", "Chat"),
    ("03", "Human evaluation", "Enter your own judgment and explanation.", "Chat"),
    (
        "04",
        "LLM judges",
        "Ask a model to judge an answer against source facts.",
        "Chat",
    ),
    (
        "05",
        "Model benchmarking",
        "Compare two models on the same three questions.",
        "Chat",
    ),
    ("06", "Chat", "Check whether a second turn remembers the product.", "Chat"),
    ("07", "RAG", "Find the right policy, then answer using it.", "RAG"),
    ("08", "Tool calling", "Let the model select and call a product lookup.", "Tools"),
    ("09", "MCP", "Call product lookup through a real MCP connection.", "MCP"),
    (
        "10",
        "Skills",
        "Select a cancellation procedure only when relevant.",
        "Procedure",
    ),
    (
        "11",
        "Single agent",
        "Inspect and cancel an eligible sandbox order.",
        "Single agent",
    ),
    (
        "12",
        "Multi-agent",
        "Compare one agent with two cooperating roles.",
        "Multi-agent",
    ),
    ("13", "Security", "Block another customer's order at the tool boundary.", "Tools"),
    (
        "14",
        "Statistical reliability",
        "Calculate counts, coverage and an interval.",
        "Statistics",
    ),
    ("15", "Operations", "Measure actual calls, tokens and latency.", "Chat"),
    ("16", "Regression / CI", "Catch a deliberately wrong price.", "Regression"),
]


class JudgeVerdict(Contract):
    verdict: Literal["correct", "incorrect", "uncertain"]
    reason: Nonblank


class DemoResult(Contract):
    run_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    topic_id: str = Field(pattern=r"^(0[1-9]|1[0-6])$")
    title: str
    system_mode: str
    evidence_kind: Literal["live", "controlled", "replay"]
    created_at_utc: str
    question: str
    source_facts: dict
    expected: str
    output: object
    status: Literal["pass", "fail", "needs_review", "not_applicable", "execution_error"]
    explanation: str
    limitation: str
    checks: list[dict]
    trace: list[dict]
    usage: dict
    provenance: dict


def catalog_check(answer: str, product_id: str, kind: str = "price") -> dict:
    product = Product.model_validate(Shop().get_product(product_id))
    return check_catalog(answer, product, kind).model_dump()


def wilson(passed: int, total: int) -> list[float] | None:
    if total < 0 or passed < 0 or passed > total:
        raise ValueError("Counts must satisfy 0 <= passed <= total")
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = passed / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
        / denominator
    )
    return [max(0.0, centre - half), min(1.0, centre + half)]


def regression_check(*, broken: bool = False) -> dict:
    answers = [
        ("SKU-NOTEBOOK", "INR 150" if broken else "INR 100"),
        ("SKU-BACKPACK", "INR 800"),
    ]
    checks = [
        {"product_id": product, "answer": answer, **catalog_check(answer, product)}
        for product, answer in answers
    ]
    return {
        "passed": all(c["status"] == "pass" for c in checks),
        "checks": checks,
        "evidence_kind": "controlled",
        "variant": "broken" if broken else "baseline",
    }


def run_demo(
    topic_id: str,
    *,
    model: str = "gemma4:e2b",
    secondary_model: str = "gemma2:2b",
    judge_model: str = "gemma4:e2b",
    provider: str = "ollama",
) -> DemoResult:
    topic = next((t for t in TOPICS if t[0] == topic_id), None)
    if topic is None:
        raise ValueError("Unknown topic")
    runtime = Runtime(model, max_calls=6 if topic_id == "05" else 4, provider=provider)
    result = dict(
        run_id=uuid.uuid4().hex,
        topic_id=topic_id,
        title=topic[1],
        system_mode=topic[3],
        evidence_kind="controlled",
        created_at_utc=datetime.now(UTC).isoformat(),
        question=topic[2],
        source_facts=snapshot(),
        expected="Inspect the stated check.",
        output=None,
        status="needs_review",
        explanation="",
        limitation="Starter demo; full module gate remains separate.",
        checks=[],
        trace=[],
        usage={},
        provenance={},
    )
    extra_events = []
    extra_models = {}
    try:
        if topic_id in {"01", "02", "03"}:
            question = (
                "How much does the backpack cost?"
                if topic_id != "03"
                else "How much does the notebook cost?"
            )
            answer = runtime.ask(question)
            result.update(
                question=question,
                expected="INR 800" if topic_id != "03" else "INR 100",
                output=answer,
            )
            if topic_id == "01":
                result["checks"] = [
                    {
                        "exact": exact_match(answer, "INR 800"),
                        "normalized": normalized_match(answer, "INR 800"),
                    }
                ]
                result.update(
                    status="pass" if answer == "INR 800" else "needs_review",
                    explanation=(
                        "String equality checks wording. A different sentence "
                        "can still express the right price."
                    ),
                    limitation=(
                        "A string mismatch is not automatically a factual error."
                    ),
                )
            elif topic_id == "02":
                check = catalog_check(answer, "SKU-BACKPACK")
                result["checks"] = [check]
                result.update(
                    status="needs_review"
                    if check["status"] == "review"
                    else check["status"],
                    explanation=(
                        "Parse a supported amount/currency and compare with "
                        "catalog 800."
                    ),
                    limitation=(
                        "Unfamiliar wording needs review. Controlled INR 900 "
                        "fails; extra claims are not substring-accepted."
                    ),
                )
            else:
                result.update(
                    explanation=(
                        "Read the facts and enter your own label and reason below."
                    ),
                    limitation=(
                        "One person's annotation is not independent reviewer "
                        "agreement or proof of mastery."
                    ),
                )
        elif topic_id == "04":
            question = "Do you ship internationally?"
            answer = runtime.ask(question)
            judgment = runtime.infer(
                [
                    {
                        "role": "system",
                        "content": "Judge only source support. "
                        "No international shipping "
                        "policy is provided. "
                        "Saying shipping is unavailable invents a policy; "
                        "admitting missing information is correct. Return JSON.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": question,
                                "answer": answer,
                                "source_policies": snapshot()["policies"],
                            }
                        ),
                    },
                ],
                role="Judge",
                model=judge_model,
                schema=JudgeVerdict.model_json_schema(),
            )
            grade = JudgeVerdict.model_validate_json(judgment)
            result.update(
                question=question,
                expected="Admit shipping policy information is missing.",
                output={"answer": answer, "judge": grade.model_dump()},
                checks=[{"judge_verdict": grade.verdict}],
                explanation=(
                    "The judge is another fallible model. Compare its "
                    "reason with your own assessment."
                ),
                limitation=(
                    "This judge has not been calibrated against "
                    "independent human labels."
                ),
            )
            result["provenance"]["judge_model"] = judge_model
            result["provenance"]["judge_is_subject_model"] = judge_model == model
        elif topic_id == "05":
            if model == secondary_model:
                raise ValueError("Select two different model IDs for this comparison")
            questions = [
                ("How much does the notebook cost?", "SKU-NOTEBOOK", "price"),
                ("Is the water bottle in stock?", "SKU-BOTTLE", "stock"),
                ("Do you ship internationally?", None, None),
            ]
            rows = []
            for selected in (model, secondary_model):
                for question, product, kind in questions:
                    messages = [
                        {
                            "role": "system",
                            "content": "Answer briefly using only these shop facts. "
                            "Admit missing information. "
                            + json.dumps(
                                {
                                    k: snapshot()[k]
                                    for k in ("products", "policies", "currency")
                                }
                            ),
                        },
                        {"role": "user", "content": question},
                    ]
                    answer = runtime.infer(messages, model=selected)
                    rows.append(
                        {
                            "model": selected,
                            "reported_model": runtime.events[-1]["completion"]["model"],
                            "question": question,
                            "answer": answer,
                            "check": catalog_check(answer, product, kind)
                            if product
                            else {
                                "status": "review",
                                "reason": "Human source-support review required",
                            },
                        }
                    )
            result.update(
                output=rows,
                checks=[{"calls": len(rows), "same_questions": True}],
                explanation=(
                    "Compare the same three tasks, scoped catalog verdicts "
                    "and observed usage."
                ),
                limitation=(
                    "Six development responses do not establish a winning "
                    "model; shipping remains ungraded. A model router may "
                    "select a different model for each request."
                ),
            )
        elif topic_id == "06":
            first = "I am asking about the backpack. Please remember the product."
            reply = runtime.ask(first)
            answer = runtime.ask(
                "How much does it cost?",
                history=[
                    {"role": "user", "content": first},
                    {"role": "assistant", "content": reply},
                ],
            )
            check = catalog_check(answer, "SKU-BACKPACK")
            result.update(
                question="Two turns: remember the backpack, then answer its price.",
                expected="The second answer reports the backpack price, INR 800.",
                output={"turn_1": reply, "turn_2": answer},
                checks=[check],
                status="needs_review"
                if check["status"] == "review"
                else check["status"],
                explanation=(
                    "The second request uses 'it'; the price check compares "
                    "the answer with the backpack price."
                ),
                limitation=(
                    "A price match alone does not prove how the reference was "
                    "resolved or establish general conversational memory."
                ),
            )
        elif topic_id == "07":
            question = "When can I cancel an order?"
            context = retrieve(question)
            wrong = retrieve(question, wrong_document=True)
            answer = runtime.ask(
                question + " Cite the document ID in brackets.", context=context
            )
            found = context["document_id"] == "cancellation"
            cited = "[cancellation]" in answer.lower()
            result.update(
                question=question,
                expected="Retrieve cancellation policy and reference it.",
                output={
                    "retrieved": context,
                    "answer": answer,
                    "controlled_wrong_retrieval": wrong,
                },
                checks=[
                    {
                        "hit_at_1": int(found),
                        "citation_marker_present": cited,
                        "controlled_wrong_hit_at_1": 0,
                    }
                ],
                status="needs_review" if found else "fail",
                explanation=(
                    "Retrieval relevance and generated-answer support are "
                    "separate checks."
                ),
                limitation=(
                    "Keyword retrieval over three documents is "
                    "introductory; a citation marker alone does not prove "
                    "claim support."
                ),
            )
        elif topic_id == "08":
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_product",
                        "description": "Look up a DemoShop product by exact ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {"product_id": {"type": "string"}},
                            "required": ["product_id"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            messages = [
                {
                    "role": "system",
                    "content": "Use the product tool for the notebook (SKU-NOTEBOOK). "
                    "Answer only from its actual result.",
                },
                {
                    "role": "user",
                    "content": "Look up the price of product SKU-NOTEBOOK (Notebook).",
                },
            ]
            selection = json.loads(
                runtime.infer(messages, role="Tool caller", tools=tools)
            )
            calls = selection.get("tool_calls", [])
            function = calls[0].get("function", {}) if len(calls) == 1 else {}
            valid = function.get("name") == "get_product" and function.get(
                "arguments"
            ) == {"product_id": "SKU-NOTEBOOK"}
            output = {"error": "No single correct native tool call; not executed"}
            final_answer = None
            if valid:
                output = runtime.tool(
                    Action(tool="get_product", product_id="SKU-NOTEBOOK")
                )
                final_message = json.loads(
                    runtime.infer(
                        [
                            *messages,
                            selection,
                            {
                                "role": "tool",
                                "tool_name": "get_product",
                                "content": json.dumps(output),
                            },
                        ],
                        role="Tool answer",
                        tools=tools,
                    )
                )
                final_answer = final_message.get("content", "")
                if final_message.get("tool_calls"):
                    final_answer = ""
            fact_correct = valid and output == Shop().get_product("SKU-NOTEBOOK")
            answer_check = (
                catalog_check(final_answer, "SKU-NOTEBOOK")
                if final_answer
                else {"status": "review", "reason": "No final text answer"}
            )
            result.update(
                output={
                    "model_selection": {
                        "tool": function.get("name"),
                        "arguments": function.get("arguments"),
                    },
                    "native_message": selection,
                    "actual_tool_result": output,
                    "final_answer": final_answer,
                },
                expected="get_product(SKU-NOTEBOOK) returns price_inr 100.",
                checks=[
                    {"selection_correct": valid, "returned_fact_correct": fact_correct},
                    answer_check,
                ],
                status=(
                    "fail"
                    if not fact_correct
                    else "needs_review"
                    if answer_check["status"] == "review"
                    else answer_check["status"]
                ),
                explanation=(
                    f"{provider} returns a native tool_calls message. "
                    "The host validates "
                    "and executes it, sends a tool-role result, "
                    "and checks the final answer."
                ),
                limitation=(
                    "Requires a tool-capable selected model. Missing or incorrect "
                    "tool calls fail visibly; no JSON-action fallback is used."
                ),
            )
        elif topic_id == "09":
            output = asyncio.run(mcp_lookup())
            runtime.events.extend(
                [
                    {
                        "kind": "tool",
                        "tool": "get_product",
                        "transport": "mcp/stdio",
                        "arguments": {"product_id": "SKU-NOTEBOOK"},
                        "result": output["response"],
                    },
                    {
                        "kind": "tool",
                        "tool": "get_product",
                        "transport": "mcp/stdio",
                        "arguments": {"wrong_argument": "SKU-NOTEBOOK"},
                        "result": output["invalid_arguments"],
                    },
                ]
            )
            good = output["response"]
            product = json.loads(
                next(c["text"] for c in good["content"] if c["type"] == "text")
            )
            valid = (
                product == Shop().get_product("SKU-NOTEBOOK")
                and output["invalid_arguments"]["is_error"]
            )
            result.update(
                output=output,
                expected=(
                    "Real stdio lookup equals direct lookup; malformed "
                    "arguments are rejected."
                ),
                status="pass" if valid else "fail",
                checks=[{"protocol_contract": bool(valid)}],
                explanation=(
                    "A separate SDK server is discovered and called "
                    "through an MCP client."
                ),
                limitation=(
                    "Local protocol demonstration; not production OAuth or "
                    "enterprise authorization validation."
                ),
            )
        elif topic_id == "10":
            positive = skill_for("Please cancel my order DS-1001")
            negative = skill_for("What is the notebook price?")
            if positive is None:
                raise ValueError("Cancellation skill was not selected")
            execution = run_agent(runtime, skill=positive)
            cancelled = execution["final_order"]["status"] == "cancelled"
            skill_in_prompt = any(
                positive["body"] in m["content"]
                for e in runtime.events
                if e["kind"] == "model"
                for m in e["messages"]
            )
            result.update(
                question="Please cancel my order DS-1001",
                output={
                    "matching_request": positive,
                    "nonmatching_request": negative,
                    "execution": execution,
                },
                expected=(
                    "Load the cancellation SKILL.md into the prompt and cancel "
                    "the owned unshipped sandbox order. "
                    "Price requests load nothing."
                ),
                status="pass"
                if cancelled and negative is None and skill_in_prompt
                else "fail",
                checks=[
                    {
                        "trigger": True,
                        "false_trigger": negative is not None,
                        "skill_body_in_prompt": skill_in_prompt,
                        "final_state_success": cancelled,
                    }
                ],
                explanation=(
                    "The host reads SKILL.md from disk, injects its Markdown into "
                    "the model prompt, records the file hash "
                    "and verifies actual tool state."
                ),
                limitation=(
                    "This checks prompt inclusion and final state, not every "
                    "instruction or the complete tool trajectory."
                ),
            )
        elif topic_id == "11":
            output = run_agent(runtime)
            success = output["final_order"]["status"] == "cancelled"
            result.update(
                output=output,
                expected="Owned unshipped order DS-1001 becomes cancelled.",
                status="pass" if success else "fail",
                checks=[{"final_state_success": success}],
                explanation=(
                    "Check the actual sandbox state, not an agent saying 'done'."
                ),
                limitation=(
                    "Four model calls and three tool calls maximum; new "
                    "runs reset all orders."
                ),
            )
        elif topic_id == "12":
            single = Runtime(model, max_calls=2, provider=provider)
            multi = Runtime(model, max_calls=2, provider=provider)
            try:
                one = run_agent(single)
                two = run_agent(multi, multi=True)
            finally:
                extra_events = single.events + multi.events
                extra_models = {**single.models, **multi.models}
            success = (
                one["final_order"]["status"]
                == two["final_order"]["status"]
                == "cancelled"
            )
            result.update(
                output={"single_agent": one, "two_roles": two},
                expected=(
                    "Compare final state, actual role handoff and calls on "
                    "the same initial order."
                ),
                checks=[{"both_succeeded": success}],
                status="pass" if success else "needs_review",
                explanation=(
                    "Two real model roles hand off policy review to order "
                    "execution; each variant starts fresh."
                ),
                limitation=(
                    "Two model calls per variant (four total). Delegation "
                    "is not presumed better; limited budgets can fail."
                ),
            )
        elif topic_id == "13":
            before = json.dumps(runtime.shop.data, sort_keys=True)
            denied = runtime.tool(Action(tool="get_order", order_id="DS-1003"))
            blocked_cancel = runtime.tool(
                Action(tool="cancel_order", order_id="DS-1003")
            )
            unchanged = before == json.dumps(runtime.shop.data, sort_keys=True)
            own = runtime.tool(Action(tool="get_order", order_id="DS-1001"))
            safe = (
                "error" in denied
                and "error" in blocked_cancel
                and unchanged
                and own.get("customer_id") == "C001"
            )
            result.update(
                output={
                    "cross_customer_read": denied,
                    "cross_customer_cancel": blocked_cancel,
                    "legitimate_read": own,
                    "state_unchanged": unchanged,
                },
                expected=(
                    "Caller C001 cannot read/change C002's order; "
                    "legitimate access still works."
                ),
                checks=[{"authorization_boundary": safe}],
                status="pass" if safe else "fail",
                explanation=(
                    "Tool-side ownership enforcement does not depend on a "
                    "model refusing."
                ),
                limitation=(
                    "Synthetic trusted caller context, not production "
                    "identity or comprehensive security testing."
                ),
            )
        elif topic_id == "14":
            answers = ["INR 100", "INR 100", "INR 150", "INR 100", "INR 200"]
            checks = [catalog_check(a, "SKU-NOTEBOOK") for a in answers]
            passed = sum(c["status"] == "pass" for c in checks)
            result.update(
                output={
                    "authored_answers": answers,
                    "passes": passed,
                    "scored": 5,
                    "pass_rate": passed / 5,
                    "wilson_95": wilson(passed, 5),
                },
                expected="3/5 = 60%; a small sample gives a wide interval.",
                checks=checks,
                status="pass",
                explanation=(
                    "Wilson's binomial formula demonstrates uncertainty, "
                    "not just a headline percentage."
                ),
                limitation=(
                    "Constructed repeated controls are not independent "
                    "sampled tasks. This interval is illustrative, not a "
                    "valid model-accuracy estimate."
                ),
            )
        elif topic_id == "15":
            answer = runtime.ask("How much does the notebook cost?")
            measured = runtime.usage()
            result.update(
                output={"answer": answer, "observed_usage": measured},
                expected=(
                    "Show actual call latency and token counts; distinguish "
                    "reported cost from missing measurements."
                ),
                status="needs_review"
                if measured["missing_token_measurements"]
                else "pass",
                explanation=(
                    "Read usage from the actual provider response; do not "
                    "invent token or cost values."
                ),
                limitation=(
                    "One call includes load/cache effects and cannot "
                    "establish throughput or production p95."
                ),
            )
        elif topic_id == "16":
            baseline, broken = regression_check(), regression_check(broken=True)
            caught = baseline["passed"] and not broken["passed"]
            result.update(
                output={"baseline": baseline, "controlled_regression": broken},
                expected=(
                    "Baseline passes; intentionally wrong notebook price is caught."
                ),
                status="pass" if caught else "fail",
                checks=[{"regression_caught": caught}],
                explanation=(
                    "The checker detects a known defect. `lab --check "
                    "--broken` exits nonzero."
                ),
                limitation=(
                    "A local CI-ready check, not evidence of a configured "
                    "remote CI pipeline."
                ),
            )
    except Exception as exc:
        result.update(
            status="execution_error",
            output={"error": str(exc)},
            explanation=(
                "The demo could not complete. No replay or synthetic "
                "response was substituted."
            ),
        )
    runtime.events.extend(extra_events)
    runtime.models.update(extra_models)
    result["trace"] = runtime.events
    result["usage"] = runtime.usage()
    result["evidence_kind"] = "live" if result["usage"]["model_calls"] else "controlled"
    result["provenance"].update(
        {
            "application_version": "browser-starter-1.2.0",
            "fixture_version": "1.0.0",
            "fixture_sha256": sha256(FIXTURE.read_bytes()),
            "models": runtime.models,
            "provider": provider,
            "sampling": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": 2048,
                "max_tokens": 256,
            },
            "python_packages": {
                name: version(name) for name in ("pydantic", "streamlit", "mcp")
            },
            "source_sha256": {
                p.relative_to(ROOT).as_posix(): sha256(p.read_bytes())
                for p in (
                    Path(__file__),
                    ROOT / "intro/src/ai_quality_evals/lab.py",
                    ROOT / "intro/src/ai_quality_evals/lab_systems.py",
                    ROOT / "intro/src/ai_quality_evals/mcp_server.py",
                    SKILL,
                    ROOT / "intro/src/ai_quality_evals/providers/ollama.py",
                    ROOT / "intro/src/ai_quality_evals/providers/openrouter.py",
                    ROOT / "intro/src/ai_quality_evals/evaluators/catalog.py",
                    ROOT / "intro/src/ai_quality_evals/evaluators/matching.py",
                    ROOT / "intro/src/ai_quality_evals/foundations.py",
                    ROOT / "pyproject.toml",
                    ROOT / "uv.lock",
                )
            },
            "git": git_provenance(ROOT),
            "full_module_validated": False,
        }
    )
    return DemoResult.model_validate(result)


def save_result(result: DemoResult) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{result.topic_id}-{result.run_id}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(result.model_dump_json(indent=2) + "\n")
    return path


def saved_runs(topic_id: str) -> list[Path]:
    return sorted(
        REPORTS.glob(f"{topic_id}-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def replay(path: Path) -> DemoResult:
    original = DemoResult.model_validate_json(path.read_bytes())
    return original.model_copy(
        update={
            "evidence_kind": "replay",
            "provenance": {
                **original.provenance,
                "original_evidence_kind": original.evidence_kind,
                "replay_source_sha256": sha256(path.read_bytes()),
            },
        }
    )


def save_human_review(
    result: DemoResult, reviewer: str, label: str, reason: str
) -> Path:
    if (
        not reviewer.strip()
        or not reason.strip()
        or label not in {"correct", "incorrect", "uncertain"}
    ):
        raise ValueError("Enter reviewer name, a real label and your explanation")
    path = REPORTS / "reviews" / f"{result.run_id}-{uuid.uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": result.run_id,
                "reviewer": reviewer.strip(),
                "label": label,
                "reason": reason.strip(),
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source": "user_entered",
                "reviewed_result_sha256": sha256(result.model_dump_json().encode()),
                "reviewed_evidence_kind": result.evidence_kind,
                "independent_agreement_established": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
