"""Offline M01 validation, comparison and reproducible JSON reporting."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
)

from ai_quality_evals.evaluators.matching import (
    RULE_VERSION,
    exact_match,
    normalize,
    normalized_match,
)

Nonblank = Annotated[str, Field(min_length=1, pattern=r"\S")]
Binary = Annotated[int, Field(ge=0, le=1)]


class Contract(BaseModel):
    """Reject coercions and misspelled fields at the evaluation boundary."""

    model_config = ConfigDict(strict=True, extra="forbid")


class Scores(Contract):
    exact: Binary
    normalized: Binary


class GoldenCase(Contract):
    case_id: Nonblank
    instruction: Nonblank
    reference: str
    candidate: str
    expected: Scores
    task_correct: bool
    note: Nonblank


class Dataset(Contract):
    dataset_id: Nonblank
    version: Nonblank
    provenance: Nonblank
    application: Literal["DemoShop"]
    application_stage: Literal["authored_fixtures"]
    fixture_version: Nonblank
    shop_snapshot: dict[str, JsonValue] = Field(min_length=1)
    cases: list[GoldenCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def unique_ids(cls, cases: list[GoldenCase]) -> list[GoldenCase]:
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique")
        return cases


class CaseResult(Contract):
    case: GoldenCase
    normalized_candidate: str
    normalized_reference: str
    scores: Scores
    golden_agreement: dict[str, bool]
    reasons: dict[str, str]
    oracle_diagnostics: dict[str, Literal["tp", "fp", "fn", "tn"]]


class Metrics(Contract):
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    golden_agreement_count: int = Field(ge=0)
    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tn: int = Field(ge=0)
    oracle_accuracy: float = Field(ge=0, le=1)


class Evaluation(Contract):
    total: int = Field(ge=1)
    results: list[CaseResult]
    metrics: dict[str, Metrics]
    disagreement_ids: list[str]
    normalized_only_passes: int
    exact_only_passes: int


class Report(Contract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str
    generated_at_utc: str
    provenance: dict[str, JsonValue]
    evaluation: Evaluation


def score_case(case: GoldenCase) -> CaseResult:
    scores = Scores(
        exact=exact_match(case.candidate, case.reference),
        normalized=normalized_match(case.candidate, case.reference),
    )
    diagnostics = {}
    for rule in ("exact", "normalized"):
        passed = bool(getattr(scores, rule))
        diagnostics[rule] = (
            ("tp" if case.task_correct else "fp")
            if passed
            else ("fn" if case.task_correct else "tn")
        )
    return CaseResult(
        case=case,
        normalized_candidate=normalize(case.candidate),
        normalized_reference=normalize(case.reference),
        scores=scores,
        golden_agreement={
            rule: getattr(scores, rule) == getattr(case.expected, rule)
            for rule in ("exact", "normalized")
        },
        reasons={
            rule: "match" if getattr(scores, rule) else f"{rule}_mismatch"
            for rule in ("exact", "normalized")
        },
        oracle_diagnostics=diagnostics,
    )


def evaluate(dataset: Dataset) -> Evaluation:
    results = [score_case(case) for case in dataset.cases]
    total = len(results)
    metrics = {}
    for rule in ("exact", "normalized"):
        passed = sum(getattr(result.scores, rule) for result in results)
        counts = {
            label: sum(result.oracle_diagnostics[rule] == label for result in results)
            for label in ("tp", "fp", "fn", "tn")
        }
        metrics[rule] = Metrics(
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total,
            golden_agreement_count=sum(r.golden_agreement[rule] for r in results),
            oracle_accuracy=(counts["tp"] + counts["tn"]) / total,
            **counts,
        )
    return Evaluation(
        total=total,
        results=results,
        metrics=metrics,
        disagreement_ids=[
            r.case.case_id for r in results if r.scores.exact != r.scores.normalized
        ],
        normalized_only_passes=sum(
            r.scores.normalized == 1 and r.scores.exact == 0 for r in results
        ),
        exact_only_passes=sum(
            r.scores.exact == 1 and r.scores.normalized == 0 for r in results
        ),
    )


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git_provenance(root: Path) -> dict[str, JsonValue]:
    """Read Git without permitting optional index updates; never create metadata."""
    if not (root / ".git").exists():
        return {"revision": None, "dirty": None, "reason": "no_git_metadata"}
    try:
        revision = subprocess.run(
            ["git", "--no-optional-locks", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return {"revision": revision, "dirty": bool(status), "reason": None}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "reason": "git_read_failed"}


def build_report(dataset_path: Path, root: Path) -> Report:
    # Hash and validate the very same bytes, avoiding dataset provenance races.
    raw = dataset_path.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    dataset_location = dataset_path.resolve()
    if dataset_location.is_relative_to(root.resolve()):
        dataset_location = dataset_location.relative_to(root.resolve())
    sources = sorted(
        path
        for directory in (
            "intro/src/ai_quality_evals",
            "intro/tests",
        )
        for path in (root / directory).rglob("*.py")
    ) + [root / "pyproject.toml", root / "uv.lock"]
    provenance = {
        "module": "01_foundations",
        "application": dataset.application,
        "application_stage": dataset.application_stage,
        "fixture_version": dataset.fixture_version,
        "shop_snapshot": dataset.shop_snapshot,
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "dataset_provenance": dataset.provenance,
        "dataset_sha256": sha256(raw),
        "rule_version": RULE_VERSION,
        "source_sha256": {
            path.relative_to(root).as_posix(): sha256(path.read_bytes())
            for path in sources
        },
        "git": git_provenance(root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pydantic": version("pydantic"),
            "pytest": version("pytest"),
            "ruff": version("ruff"),
            "package": version("ai-quality-evals"),
        },
        "inference": {
            "provider": None,
            "model": None,
            "judge": None,
            "prompt_version": None,
            "sampling": None,
            "seed": None,
            "tokens": None,
            "cost": None,
            "reason": "authored DemoShop outputs; no inference or prompt execution",
        },
        "reproduction_command": (
            "uv run --locked --offline python -m ai_quality_evals.foundations "
            f'--dataset "{dataset_location.as_posix()}" '
            "--output reports/01_foundations/run-1.json"
        ),
    }
    evaluation = evaluate(dataset)
    stable = {"provenance": provenance, "evaluation": evaluation.model_dump()}
    digest = sha256(json.dumps(stable, sort_keys=True, ensure_ascii=True).encode())
    return Report(
        run_id=digest,
        generated_at_utc=datetime.now(UTC).isoformat(),
        provenance=provenance,
        evaluation=evaluation,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args.dataset, Path.cwd())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, ValidationError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(f"Wrote {args.output}; run_id={report.run_id}")
    return (
        0
        if all(
            m.golden_agreement_count == report.evaluation.total
            for m in report.evaluation.metrics.values()
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
