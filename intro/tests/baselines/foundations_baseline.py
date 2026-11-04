"""Standard-library execution path for the trusted M01 synthetic fixture."""

import argparse
import json
from pathlib import Path

from ai_quality_evals.evaluators.matching import exact_match, normalized_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    results = [
        {
            "case_id": case["case_id"],
            "exact": exact_match(case["candidate"], case["reference"]),
            "normalized": normalized_match(case["candidate"], case["reference"]),
        }
        for case in dataset["cases"]
    ]
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
