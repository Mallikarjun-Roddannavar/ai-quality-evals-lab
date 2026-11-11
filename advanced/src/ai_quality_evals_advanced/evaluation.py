"""A2: distinguish mechanical agreement, authored truth and human review."""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from ai_quality_evals.evaluators.matching import exact_match, normalized_match
from ai_quality_evals_advanced.contracts import ROOT, Contract

DATASET = ROOT / "advanced/datasets/evaluation-v1.json"


class Case(Contract):
    id: str
    request: str
    reference: str
    answer: str
    label: bool
    category: str
    split: Literal["development", "held_out"]


class Annotation(Contract):
    case_id: str
    reviewer: str = Field(min_length=1)
    label: Literal["correct", "incorrect", "uncertain"]
    reason: str = Field(min_length=1)


def cases(split: str = "development") -> list[Case]:
    if split not in {"development", "held_out", "all"}:
        raise ValueError("Unknown split")
    dataset = json.loads(DATASET.read_bytes())
    rows = []
    case_ids = set()
    for item in dataset["cases"]:
        case = Case.model_validate(item)
        if case.id in case_ids:
            raise ValueError("Duplicate case identity")
        case_ids.add(case.id)
        if split == "all" or case.split == split:
            rows.append(case)
    return rows


def controls(split: str = "development") -> dict:
    rows = []
    for case in cases(split):
        exact = exact_match(case.answer, case.reference)
        normal = normalized_match(case.answer, case.reference)
        rows.append({**case.model_dump(), "exact": exact, "normalized": normal})
    counts = {}
    for metric in ("exact", "normalized"):
        passes = 0
        false_pass_ids = []
        false_failure_ids = []
        for row in rows:
            passes += row[metric]
            if row[metric] and not row["label"]:
                false_pass_ids.append(row["id"])
            if not row[metric] and row["label"]:
                false_failure_ids.append(row["id"])
        counts[metric] = {
            "passes": passes,
            "total": len(rows),
            "false_pass_ids": false_pass_ids,
            "false_failure_ids": false_failure_ids,
        }
    return {
        "evidence_kind": "authored_control",
        "split": split,
        "cases": rows,
        "counts": counts,
        "human_agreement": None,
        "limitation": "Authored labels are controls, not human calibration.",
    }


def export_annotations(path: Path):
    """Blind to control labels and grader results; no invented reviewer records."""
    items = []
    for case in cases("all"):
        items.append(
            {"case_id": case.id, "request": case.request, "answer": case.answer}
        )
    path.write_text(
        json.dumps(
            {
                "rubric_version": "1.0.0",
                "rubric": (
                    "Check facts, completeness and abstention. "
                    "Choose uncertain when ambiguous."
                ),
                "items": items,
                "source_facts": "intro/datasets/demoshop/foundations-v1.json",
                "annotations": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def import_annotations(path: Path) -> list[Annotation]:
    data = json.loads(path.read_bytes())
    rows = []
    for item in data["annotations"]:
        rows.append(Annotation.model_validate(item))
    identities = set()
    for annotation in rows:
        identities.add((annotation.case_id, annotation.reviewer))
    if len(identities) != len(rows):
        raise ValueError("Unknown case or duplicate reviewer annotation")
    if rows:
        valid_ids = set()
        for case in cases("all"):
            valid_ids.add(case.id)
        for annotation in rows:
            if annotation.case_id not in valid_ids:
                raise ValueError("Unknown case or duplicate reviewer annotation")
    return rows


def agreement(rows: list[Annotation], left: str, right: str) -> dict:
    if left == right:
        raise ValueError("Distinct reviewers required")
    a = {}
    b = {}
    for row in rows:
        if row.reviewer == left:
            a[row.case_id] = row.label
        if row.reviewer == right:
            b[row.case_id] = row.label
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return {"paired": 0, "agreement": None, "kappa": None}
    matches = 0
    ca, cb = Counter(), Counter()
    disagreements = []
    for key in shared:
        if a[key] == b[key]:
            matches += 1
        else:
            disagreements.append(key)
        ca[a[key]] += 1
        cb[b[key]] += 1
    observed = matches / len(shared)
    expected_count = 0
    for label in ca.keys() | cb.keys():
        expected_count += ca[label] * cb[label]
    expected = expected_count / len(shared) ** 2
    return {
        "paired": len(shared),
        "agreement": observed,
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
        "disagreements": disagreements,
        "adjudication": "requires human review",
    }


def wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    if not total:
        return None
    p = successes / total
    divisor = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / divisor
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / divisor
    return [max(0.0, centre - radius), min(1.0, centre + radius)]
