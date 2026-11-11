"""A5: paired task statistics, observed attempts and local seeded regression."""

import json
import math
import random
import statistics
from collections import defaultdict

from ai_quality_evals.evaluators.catalog import Product, check_catalog
from ai_quality_evals_advanced.contracts import ARTIFACTS
from ai_quality_evals_advanced.domain import snapshot
from ai_quality_evals_advanced.evaluation import controls, wilson


def paired_bootstrap(pairs: list[dict], samples=2000, seed=42):
    """Resample independent case clusters; keep repeated trials together."""
    groups = defaultdict(list)
    for p in pairs:
        if p["left"] not in {0, 1} or p["right"] not in {0, 1}:
            raise ValueError("Binary paired outcomes required")
        groups[p["case_id"]].append(p["right"] - p["left"])
    if not groups:
        return {"clusters": 0, "delta": None, "interval": None}
    means = [statistics.mean(v) for v in groups.values()]
    rng = random.Random(seed)
    draws = sorted(
        statistics.mean(rng.choices(means, k=len(means))) for _ in range(samples)
    )
    return {
        "clusters": len(means),
        "trials": len(pairs),
        "delta": statistics.mean(means),
        "interval": [
            draws[int(samples * 0.025)],
            draws[min(samples - 1, int(samples * 0.975))],
        ],
        "resampling_unit": "case_id cluster",
        "seed": seed,
        "draws": samples,
    }


def statistics_report(split="development"):
    result = controls(split)
    pairs = [
        {"case_id": r["id"], "left": r["exact"], "right": r["normalized"]}
        for r in result["cases"]
    ]
    return {
        "matched_comparison": paired_bootstrap(pairs),
        "wilson": {
            m: wilson(c["passes"], c["total"]) for m, c in result["counts"].items()
        },
        "counts": result["counts"],
        "split": split,
        "evidence_kind": "authored_control",
        "limitation": (
            "Curated controls are not a random population sample. "
            "Intervals illustrate assumptions, not general accuracy."
        ),
        "human_agreement": None,
    }


def operation_summary(attempts):
    times = sorted(a["elapsed_ms"] for a in attempts)

    def quantile(p):
        return times[max(0, math.ceil(p * len(times)) - 1)] if times else None

    complete = [a for a in attempts if a["status"] == "completed"]
    known = all(
        a.get("prompt_tokens") is not None and a.get("completion_tokens") is not None
        for a in complete
    )
    return {
        "calls": len(attempts),
        "errors": sum(a["status"] != "completed" for a in attempts),
        "latency_p50_ms": quantile(0.5),
        "latency_p95_ms": quantile(0.95),
        "attempt_elapsed_ms": sum(times),
        "token_coverage": sum(
            a.get("prompt_tokens") is not None
            and a.get("completion_tokens") is not None
            for a in attempts
        )
        / len(attempts)
        if attempts
        else None,
        "prompt_tokens": sum(a["prompt_tokens"] for a in complete)
        if complete and known
        else None,
        "completion_tokens": sum(a["completion_tokens"] for a in complete)
        if complete and known
        else None,
        "cost": None,
        "cost_note": "Not measured; unknown is not zero",
    }


def regression(variant="baseline"):
    if variant not in {
        "baseline",
        "wrong-price",
        "wrong-policy",
        "wrong-retrieval",
        "wrong-authorization",
    }:
        raise ValueError("Unknown controlled variant")
    from ai_quality_evals_advanced.applications import retrieve

    source = snapshot()
    product = Product.model_validate(
        next(p for p in source["products"] if p["product_id"] == "SKU-BACKPACK")
    )
    price = 900 if variant == "wrong-price" else 800
    window = (
        30 if variant == "wrong-policy" else source["policies"]["return_window_days"]
    )
    retrieved = (
        []
        if variant == "wrong-retrieval"
        else [d["id"] for d in retrieve("Returns delivered days")]
    )
    order = next(o for o in source["orders"] if o["order_id"] == "DS-1003")
    decision = (
        True if variant == "wrong-authorization" else order["customer_id"] == "C001"
    )
    checks = {
        "price": check_catalog(f"{price} INR", product, "price").status == "pass",
        "policy": window == 14,
        "retrieval": "returns" in retrieved,
        "access_boundary": not decision,
    }
    return {
        "variant": variant,
        "checks": checks,
        "gate_pass": all(checks.values()),
        "evidence_kind": "authored_mutant"
        if variant != "baseline"
        else "deterministic_baseline",
        "limitation": "Mutation probes never install unsafe authorization in tools.",
    }


def read_evidence(identifier, base=ARTIFACTS):
    from ai_quality_evals_advanced.contracts import run_folder

    if identifier == "latest":
        paths = [*base.glob("*/experiment.json"), *base.glob("*/result.json")]
        if not paths:
            raise ValueError("No saved evidence")
        path = max(paths, key=lambda p: p.stat().st_mtime)
    else:
        folder = run_folder(identifier, base)
        path = folder / (
            "experiment.json"
            if (folder / "experiment.json").exists()
            else "result.json"
        )
    return json.loads(path.read_bytes())


def compare_saved(left, right):
    a, b = read_evidence(left), read_evidence(right)
    # Comparability is checked, not inferred from nearby timestamps.
    if a.get("experiment", a.get("case_id")) != b.get("experiment", b.get("case_id")):
        raise ValueError("Different evaluation tasks cannot be paired")
    if a.get("split") != b.get("split"):
        raise ValueError("Different splits cannot be paired")
    if a["provenance"].get("fixture_sha256", a["provenance"].get("case_sha256")) != b[
        "provenance"
    ].get("fixture_sha256", b["provenance"].get("case_sha256")):
        raise ValueError("Case or fixture provenance differs")
    return {
        "left": a["run_id"],
        "right": b["run_id"],
        "matched_task": True,
        "left_result": a.get("result", a.get("verdict")),
        "right_result": b.get("result", b.get("verdict")),
        "elapsed_delta_ms": b.get("elapsed_ms", 0) - a.get("elapsed_ms", 0),
        "limitation": "Inspect configuration/code changes before causal claims.",
    }


def operation_trials(inference):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    from ai_quality_evals_advanced.providers import Inference

    def trial(product):
        subject = Inference(
            inference.config.model_copy(update={"max_model_calls": 2}),
            inference.telemetry,
        )
        row = {"case_id": product["product_id"]}
        try:
            reply = subject.generate(
                [
                    {
                        "role": "system",
                        "content": "Answer using this catalog fact only: "
                        + json.dumps(product),
                    },
                    {"role": "user", "content": "What is the price? Include INR."},
                ]
            )
            row.update(
                output=reply.text,
                check=check_catalog(
                    reply.text, Product.model_validate(product), "price"
                ).model_dump(),
            )
        except Exception as exc:
            row["execution_error"] = type(exc).__name__
        row["attempts"] = subject.attempts
        return row

    with ThreadPoolExecutor(max_workers=inference.config.concurrency) as pool:
        futures = [
            pool.submit(copy_context().run, trial, p) for p in snapshot()["products"]
        ]
        rows = [f.result() for f in futures]
    attempts = [a for r in rows for a in r["attempts"]]
    return {
        "cases": rows,
        "summary": operation_summary(attempts),
        "concurrency": inference.config.concurrency,
        "sampling": "three distinct product cases, one trial each",
    }
