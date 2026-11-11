"""Versioned 30-scenario application experiment; expectations never enter prompts."""

import json
from typing import Literal

from ai_quality_evals.evaluators.catalog import Product, check_catalog
from ai_quality_evals_advanced.agents import agent_run
from ai_quality_evals_advanced.analysis import operation_summary
from ai_quality_evals_advanced.applications import chat, rag
from ai_quality_evals_advanced.contracts import ROOT, Contract
from ai_quality_evals_advanced.domain import snapshot
from ai_quality_evals_advanced.providers import Inference
from ai_quality_evals_advanced.runtime import atomic_json


class Scenario(Contract):
    id: str
    mode: Literal["chat", "rag", "single", "multi"]
    turns: list[str]
    actor: Literal["C001", "C002"]
    expectation: dict
    split: Literal["development", "held_out"]


def scenarios(split):
    if split not in {"development", "held_out", "all"}:
        raise ValueError("Unknown split")
    dataset = json.loads((ROOT / "advanced/datasets/scenarios-v1.json").read_bytes())
    rows = [Scenario.model_validate(r) for r in dataset["cases"]]
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate scenario ID")
    return [r for r in rows if split == "all" or r.split == split]


def run_batch(config, telemetry, folder, split="development"):
    results = []
    products = {
        p["product_id"]: Product.model_validate(p) for p in snapshot()["products"]
    }
    for case in scenarios(split):
        trial = folder / case.id
        trial.mkdir()
        subject = Inference(config.model_copy(update={"max_model_calls": 4}), telemetry)
        row = {
            "case_id": case.id,
            "split": case.split,
            "mode": case.mode,
            "request": case.turns,
            "execution_status": "completed",
            "verdict": "needs_review",
        }
        try:
            with telemetry.span("case." + case.id, case_id=case.id):
                if case.mode == "chat":
                    execution = chat(subject, case.turns)
                    actual = execution["outputs"][-1]
                    expected = case.expectation
                    if expected["kind"] in {"price", "availability"}:
                        check = check_catalog(
                            actual,
                            products[expected["product_id"]],
                            "stock" if expected["kind"] == "availability" else "price",
                        ).model_dump()
                        row.update(
                            verdict="needs_review"
                            if check["status"] == "review"
                            else check["status"],
                            check=check,
                        )
                elif case.mode == "rag":
                    execution = rag(
                        subject,
                        case.turns[0],
                        retrieval=config.retrieval,
                        embedding_model=config.embedding_model,
                    )
                    relevant = case.expectation["relevant_doc"]
                    row["retrieval_hit"] = (
                        None
                        if relevant is None
                        else relevant in {d["id"] for d in execution["retrieved"]}
                    )
                    # Citation validity/retrieval hit alone cannot grant answer support.
                else:
                    execution = agent_run(
                        subject,
                        trial,
                        case.turns[0],
                        actor=case.actor,
                        variant="langgraph",
                        multi=case.mode == "multi",
                    )
                    if execution["error"]:
                        row["execution_status"] = "execution_error"
                    expected = case.expectation
                    if expected["order_id"]:
                        correct = (
                            execution["final_state"][expected["order_id"]]["status"]
                            == expected["status"]
                        )
                    else:
                        correct = all(
                            execution["final_state"][o["order_id"]]["status"]
                            == o["status"]
                            for o in snapshot()["orders"]
                        )
                    row["verdict"] = "pass" if correct else "fail"
                    row["outcome_pass"] = correct
                    if execution["error"]:
                        row["verdict"] = "execution_error"
                    row["criterion"] = (
                        "sandbox outcome; response/trajectory require separate review"
                    )
                row["execution"] = execution
        except Exception as exc:
            row.update(
                execution_status="execution_error",
                verdict="execution_error",
                error=type(exc).__name__,
            )
        row["attempts"] = subject.attempts
        row["provider_metadata"] = subject.metadata
        row["expectation"] = case.expectation
        atomic_json(trial / "case.json", row)
        results.append(row)
    counts = {
        label: sum(r["verdict"] == label for r in results)
        for label in ("pass", "fail", "needs_review", "execution_error")
    }
    return {
        "cases": results,
        "counts": counts,
        "graded_coverage": sum(r["verdict"] in {"pass", "fail"} for r in results)
        / len(results),
        "operations": operation_summary([a for r in results for a in r["attempts"]]),
        "limitation": "Narrow criteria; policy/RAG semantic review remains pending.",
    }
