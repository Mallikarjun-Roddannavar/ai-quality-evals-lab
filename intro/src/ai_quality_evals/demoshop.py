"""Minimal single-turn DemoShop assistant and recorded-output replay."""

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, field_validator

from ai_quality_evals.evaluators.matching import exact_match, normalized_match
from ai_quality_evals.foundations import (
    Contract,
    Dataset,
    Nonblank,
    git_provenance,
    sha256,
)
from ai_quality_evals.providers import ollama

DEFAULT_FIXTURE = Path("intro/datasets/demoshop/foundations-v1.json")
DEFAULT_PROMPT = Path("intro/configs/prompts/demoshop-answer-v1.txt")


class Question(Contract):
    case_id: Nonblank
    question: Nonblank
    reference: str | None = None
    expected_behavior: Nonblank
    source_paths: list[str] = Field(default_factory=list)


class QuestionSet(Contract):
    dataset_id: Nonblank
    version: Nonblank
    provenance: Nonblank
    cases: list[Question] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def unique_ids(cls, cases: list[Question]) -> list[Question]:
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("Question case IDs must be unique")
        return cases


def build_messages(
    fixture: Dataset, prompt: str, question: str
) -> list[dict[str, str]]:
    # Only public catalog/policies enter this answer-only pilot. No goldens,
    # candidate answers, order records or customer identifiers go to the model.
    snapshot = fixture.shop_snapshot
    public_data = {key: snapshot[key] for key in ("currency", "products", "policies")}
    return [
        {
            "role": "system",
            "content": prompt + "\nSHOP DATA:\n" + json.dumps(public_data),
        },
        {"role": "user", "content": question},
    ]


def run_questions(
    fixture_path: Path,
    prompt_path: Path,
    questions: QuestionSet,
    *,
    model: str,
    base_url: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    fixture_bytes = fixture_path.read_bytes()
    fixture = Dataset.model_validate_json(fixture_bytes)
    prompt_bytes = prompt_path.read_bytes()
    prompt = prompt_bytes.decode("utf-8")
    if not prompt.strip():
        raise ValueError("Prompt must not be blank")
    model_info = ollama.installed_model(base_url, model)
    started_at = datetime.now(UTC).isoformat()
    results = []
    for case in questions.cases:
        messages = build_messages(fixture, prompt, case.question)
        record = {
            "case": case.model_dump(),
            "messages": messages,
            "completion": None,
            "matching_scores": None,
            "error": None,
            "manual_review": "pending; matching scores are not factual correctness",
        }
        try:
            completion = ollama.chat(
                base_url, model, messages, max_tokens=max_tokens, timeout=timeout
            )
            record["completion"] = completion.model_dump()
            if case.reference is not None:
                record["matching_scores"] = {
                    "exact": exact_match(completion.output, case.reference),
                    "normalized": normalized_match(completion.output, case.reference),
                }
        except (RuntimeError, ValueError, KeyError) as exc:
            record["error"] = str(exc)
        results.append(record)
    root = Path.cwd()
    source_paths = [
        Path(__file__).resolve(),
        Path(ollama.__file__).resolve(),
        root / "intro/src/ai_quality_evals/evaluators/matching.py",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    return {
        "schema_version": "1.0.0",
        "mode": "live",
        "application": "DemoShop",
        "application_version": "answering-pilot-1.0.0",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "provider": "ollama",
            "base_url": base_url,
            "requested_model": model,
            "installed_model": model_info,
            "sampling": {
                "temperature": 0,
                "seed": 42,
                "num_predict": max_tokens,
                "num_ctx": 2048,
            },
            "timeout_seconds": timeout,
            "fixture_version": fixture.fixture_version,
            "fixture_sha256": sha256(fixture_bytes),
            "prompt_file": str(prompt_path),
            "prompt_version": "1.0.0",
            "prompt_sha256": sha256(prompt_bytes),
            "questions": questions.model_dump(),
            "questions_sha256": sha256(questions.model_dump_json().encode()),
            "git": git_provenance(root),
            "python": platform.python_version(),
            "source_sha256": {
                path.relative_to(root).as_posix(): sha256(path.read_bytes())
                for path in source_paths
            },
            "cost": None,
            "cost_note": "local execution; hardware/energy cost not measured",
            "judge": None,
        },
        "summary": {
            "requested": len(results),
            "completed": sum(r["completion"] is not None for r in results),
            "errors": sum(r["error"] is not None for r in results),
            "task_correctness": (
                "not automatically graded; inspect answers against facts"
            ),
        },
        "results": results,
    }


def display(report: dict, *, replay: bool = False) -> None:
    print("RECORDED OUTPUT REPLAY; no model call" if replay else "LIVE OLLAMA OUTPUT")
    for result in report["results"]:
        print(f"\n{result['case']['case_id']}: {result['case']['question']}")
        if result["completion"] is None:
            print(f"ERROR: {result['error']}")
        else:
            print(result["completion"]["output"])
            print(f"String-match diagnostics: {result['matching_scores']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question")
    group.add_argument("--questions", type=Path)
    group.add_argument("--replay", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL"))
    parser.add_argument(
        "--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.replay is not None:
            display(json.loads(args.replay.read_text(encoding="utf-8")), replay=True)
            return 0
        if not args.model or args.output is None:
            parser.error("Live runs require --model (or OLLAMA_MODEL) and --output")
        if args.max_tokens <= 0 or args.timeout <= 0:
            parser.error("--max-tokens and --timeout must be positive")
        questions = (
            QuestionSet.model_validate_json(args.questions.read_bytes())
            if args.questions is not None
            else QuestionSet(
                dataset_id="interactive",
                version="1.0.0",
                provenance=(
                    "User-supplied single question; no authored reference answer"
                ),
                cases=[
                    Question(
                        case_id="interactive-1",
                        question=args.question,
                        expected_behavior="Inspect against supplied shop facts",
                    )
                ],
            )
        )
        report = run_questions(
            args.fixture,
            args.prompt,
            questions,
            model=args.model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        display(report)
        print(f"\nSaved {args.output}")
        return 1 if report["summary"]["errors"] else 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"DemoShop run failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
