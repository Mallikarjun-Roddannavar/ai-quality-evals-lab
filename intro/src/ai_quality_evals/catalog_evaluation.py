"""Offline catalog evaluation of recorded DemoShop answers, with explicit coverage."""

import argparse
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from ai_quality_evals.evaluators.catalog import RULE_VERSION, Product, check_catalog
from ai_quality_evals.evaluators.matching import exact_match, normalized_match
from ai_quality_evals.foundations import Contract, Dataset, Nonblank, sha256


class Rule(Contract):
    question: Nonblank
    product_id: Nonblank
    check: Literal["price", "stock"]


class RuleSet(Contract):
    dataset_id: Nonblank
    version: Nonblank
    rules: list[Rule] = Field(min_length=1)

    @field_validator("rules")
    @classmethod
    def unique_questions(cls, rules: list[Rule]) -> list[Rule]:
        if len({r.question for r in rules}) != len(rules):
            raise ValueError("Each exact question must have one rule")
        return rules


def evaluate_recordings(
    inputs: list[Path], fixture_path: Path, rules_path: Path
) -> dict:
    if len({p.resolve() for p in inputs}) != len(inputs):
        raise ValueError("Duplicate input reports would double-count responses")
    fixture_bytes = fixture_path.read_bytes()
    fixture = Dataset.model_validate_json(fixture_bytes)
    rules_bytes = rules_path.read_bytes()
    rules = RuleSet.model_validate_json(rules_bytes)
    products = [Product.model_validate(p) for p in fixture.shop_snapshot["products"]]
    catalog = {p.product_id: p for p in products}
    if len(catalog) != len(products):
        raise ValueError("Catalog product IDs must be unique")
    if fixture.shop_snapshot["currency"] != "INR":
        raise ValueError("This price-rule version supports INR only")
    mapping = {r.question: r for r in rules.rules}
    if any(r.product_id not in catalog for r in rules.rules):
        raise ValueError("A configured product is missing from the selected catalog")
    results = []
    input_hashes = {}
    for path in inputs:
        raw = path.read_bytes()
        recording = json.loads(raw)
        if (
            recording.get("application") != "DemoShop"
            or recording.get("mode") != "live"
        ):
            raise ValueError("Input must be a captured live DemoShop recording")
        if recording["provenance"]["fixture_sha256"] != sha256(fixture_bytes):
            raise ValueError("Recording fixture differs from selected source facts")
        input_hashes[path.as_posix()] = sha256(raw)
        for entry in recording["results"]:
            question = entry["case"]["question"]
            completion = entry.get("completion")
            if completion and not isinstance(completion.get("output"), str):
                raise ValueError("Captured answer must be a string")
            row = {
                "input_report": path.as_posix(),
                "case_id": entry["case"]["case_id"],
                "question": question,
                "model": recording["provenance"].get("requested_model"),
                "answer": completion["output"] if completion else None,
                "status": "not_evaluated",
                "reason": "no_catalog_rule_for_question",
                "rule": None,
                "expected": None,
                "observed": None,
                "string_matches": None,
            }
            if completion is None or entry.get("error"):
                row.update(status="error", reason="recorded_generation_error")
            elif question in mapping:
                rule = mapping[question]
                row["rule"] = rule.model_dump()
                product = catalog[rule.product_id]
                if completion.get("finish_reason") != "stop":
                    row.update(status="review", reason="completion_may_be_truncated")
                else:
                    row.update(
                        check_catalog(
                            completion["output"], product, rule.check
                        ).model_dump()
                    )
                    reference = (
                        f"INR {product.price_inr}"
                        if rule.check == "price"
                        else "in stock"
                        if product.stock > 0
                        else "out of stock"
                    )
                    row["string_matches"] = {
                        "reference": reference,
                        "exact": exact_match(completion["output"], reference),
                        "normalized": normalized_match(completion["output"], reference),
                    }
            results.append(row)
    counts = Counter(row["status"] for row in results)
    root = Path.cwd()
    sources = [
        Path(__file__).resolve(),
        root / "intro/src/ai_quality_evals/evaluators/catalog.py",
        root / "intro/src/ai_quality_evals/evaluators/matching.py",
        root / "intro/src/ai_quality_evals/foundations.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    stable = {
        "schema_version": "1.0.0",
        "evaluation": "demoshop-catalog",
        "rule_version": RULE_VERSION,
        "config_version": rules.version,
        "config_id": rules.dataset_id,
        "mode": "offline_recording_evaluation",
        "provenance": {
            "fixture_sha256": sha256(fixture_bytes),
            "rules_sha256": sha256(rules_bytes),
            "input_sha256": input_hashes,
            "source_sha256": {
                p.relative_to(root).as_posix(): sha256(p.read_bytes()) for p in sources
            },
            "python": platform.python_version(),
        },
        "summary": {
            "total": len(results),
            **{
                status: counts[status]
                for status in ("pass", "fail", "review", "not_evaluated", "error")
            },
            "note": "Catalog checks only; this is not overall assistant accuracy",
        },
        "results": results,
    }
    return {
        **stable,
        "run_id": sha256(json.dumps(stable, sort_keys=True).encode()),
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("intro/datasets/demoshop/foundations-v1.json"),
    )
    parser.add_argument(
        "--rules", type=Path, default=Path("intro/configs/evaluations/catalog-v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_recordings(args.input, args.fixture, args.rules)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Catalog evaluation failed: {exc}\n")
    print(json.dumps(report["summary"]))
    print(f"Saved {args.output}; run_id={report['run_id']}")
    # A scored model failure remains a successful evaluation execution.
    return 1 if report["summary"]["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
