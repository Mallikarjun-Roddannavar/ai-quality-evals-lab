"""DemoShop evaluation CLI: explicit execution, saved evidence and local gates."""

import argparse
import json
import os
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ai_quality_evals_advanced.contracts import Settings
from ai_quality_evals_advanced.providers import request
from ai_quality_evals_advanced.security import redact


def doctor(config: Settings, offline: bool) -> dict:
    checks = []
    for name in (
        "pydantic",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
        "arize-phoenix",
        "deepeval",
        "ragas",
        "inspect-ai",
        "langchain",
        "langgraph",
    ):
        try:
            checks.append(
                {
                    "component": name,
                    "status": "installed",
                    "version": version(name),
                    "runtime_import_verified": False,
                }
            )
        except PackageNotFoundError:
            checks.append(
                {
                    "component": name,
                    "status": "missing",
                    "setup": "uv sync --locked --extra dev --extra advanced",
                }
            )
    if not offline:
        if config.provider == "ollama":
            try:
                tags = request(config.ollama_url.rstrip("/") + "/api/tags", None, 3)
                ready = any(
                    m.get("name") == config.model for m in tags.get("models", [])
                )
                checks.append(
                    {
                        "component": "subject_model",
                        "status": "ready" if ready else "missing",
                        "model": config.model,
                        "download_attempted": False,
                    }
                )
            except RuntimeError:
                checks.append({"component": "ollama", "status": "unavailable"})
        else:
            checks.append(
                {
                    "component": "openrouter_key",
                    "status": "configured"
                    if os.environ.get("OPENROUTER_API_KEY")
                    else "missing",
                    "live_connection_verified": False,
                }
            )
        if config.export_traces:
            import http.client
            from urllib.parse import urlsplit

            url = urlsplit(config.phoenix_endpoint)
            conn = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
            try:
                conn.request("GET", "/")
                response = conn.getresponse()
                checks.append(
                    {
                        "component": "phoenix_http",
                        "status": "reachable"
                        if response.status == 200
                        else "unavailable",
                        "trace_ingestion_verified": False,
                    }
                )
            except OSError:
                checks.append({"component": "phoenix_http", "status": "unavailable"})
            finally:
                conn.close()
    return {
        "milestone": "A0-A5 engineering",
        "offline": offline,
        "checks": checks,
        "implemented_framework_trials": [
            "DeepEval",
            "Ragas",
            "Inspect",
            "LangChain",
            "LangGraph",
            "promptfoo",
        ],
        "docker_required": False,
    }


def main() -> int:
    # Model answers can contain symbols such as ₹ on Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "doctor", help="Check setup without generating an answer"
    )
    check.add_argument("--offline", action="store_true")
    commands.add_parser("validate", help="Validate current settings and the price case")
    suite = commands.add_parser("suite", help="Run a named evaluation experiment")
    suite.add_argument(
        "name",
        choices=[
            "foundations",
            "judges",
            "chat",
            "rag",
            "benchmark",
            "tools",
            "mcp",
            "mcp-selection",
            "skills",
            "retrieval",
            "agents",
            "security",
            "statistics",
            "regression",
            "operations",
            "bias",
            "application",
            "promptfoo",
        ],
    )
    suite.add_argument("--mode", choices=["offline", "live"], required=True)
    suite.add_argument("--provider", choices=["ollama", "openrouter"])
    suite.add_argument("--models", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    suite.add_argument(
        "--split", choices=["development", "held_out", "all"], default="development"
    )
    annotations = commands.add_parser(
        "annotations", help="Export blind review or validate real labels"
    )
    annotations.add_argument("action", choices=["export", "import"])
    annotations.add_argument("path", type=Path)
    calibration = commands.add_parser(
        "human-judge", help="Review live answers and compare a judge"
    )
    calibration.add_argument(
        "action", choices=["generate", "review", "judge", "compare", "trace"]
    )
    calibration.add_argument("run_id", nargs="?")
    calibration.add_argument("--model", help="OpenRouter model ID")
    calibration.add_argument("--limit", type=int, default=5)
    compare = commands.add_parser(
        "compare", help="Compare matched saved evidence; no inference"
    )
    compare.add_argument("left")
    compare.add_argument("right")
    gate = commands.add_parser(
        "regression", help="Run a local controlled regression gate"
    )
    gate.add_argument(
        "--variant",
        choices=[
            "baseline",
            "wrong-price",
            "wrong-policy",
            "wrong-retrieval",
            "wrong-authorization",
        ],
        default="baseline",
    )
    commands.add_parser(
        "phoenix", help="Run native local Phoenix in this terminal; no Docker"
    )
    run = commands.add_parser("run", help="Run the first operational price example")
    run.add_argument("--example", choices=["price"], default="price")
    run.add_argument("--mode", choices=["offline", "live"], required=True)
    for command in ("report", "replay"):
        child = commands.add_parser(
            command, help="Read saved evidence; no inference or tool execution"
        )
        child.add_argument("run_id", nargs="?", default="latest")
    args = parser.parse_args()
    try:
        from ai_quality_evals_advanced.runtime import (
            execute,
            load_case,
            load_settings,
            read_result,
        )

        config = load_settings(args.config)
        if config.provider == "openrouter":
            from ai_quality_evals.providers.openrouter import load_dotenv

            load_dotenv()
        if args.command == "human-judge":
            from ai_quality_evals_advanced import human_judge
            from ai_quality_evals_advanced.contracts import ARTIFACTS, run_folder

            if args.action == "generate":
                data = human_judge.generate(args.model, args.limit)
                print(f"Run ID: {data['run_id']}")
                generated = [
                    row
                    for row in data["cases"]
                    if row.get("answer_origin") != "authored_control"
                ]
                controls = [
                    row
                    for row in data["cases"]
                    if row.get("answer_origin") == "authored_control"
                ]
                completed = sum(row["status"] == "completed" for row in generated)
                print(
                    f"Generated {completed} of {len(generated)} model answers; "
                    f"added {len(controls)} authored control(s)."
                )
                for row in data["cases"]:
                    if row["status"] != "completed":
                        print(f"  {row['case_id']}: {row['error']}")
                print("Next: evals human-judge review " + data["run_id"])
                return (
                    0
                    if all(row["status"] == "completed" for row in data["cases"])
                    else 1
                )
            if not args.run_id:
                raise ValueError("Provide the run ID shown by generate")
            if args.action == "trace":
                summary = human_judge.trace_saved(args.run_id)
                print("Saved-answer replay trace:", summary["status"])
                print("No model calls were made.")
                return 0 if summary["status"] == "exported" else 1
            if args.action == "review":
                folder = run_folder(args.run_id, ARTIFACTS)
                data = json.loads(
                    (folder / "responses.json").read_text(encoding="utf-8")
                )
                print("Shop facts:", json.dumps(data["facts"], indent=2))
                print("Rubric:", data["rubric"])
                reviewer = input("Your name: ").strip()
                assessments = []
                for row in data["cases"]:
                    if row["status"] != "completed":
                        continue
                    if row.get("answer_origin") == "authored_control":
                        print(
                            "\nAuthored control: a prepared answer "
                            "for testing the judge."
                        )
                    print("\nQuestion:", row["question"])
                    print("Answer:", row["answer"])
                    while True:
                        label = (
                            input("Label (correct/incorrect/uncertain): ")
                            .strip()
                            .lower()
                        )
                        if label in {"correct", "incorrect", "uncertain"}:
                            break
                    reason = input("Why? ").strip()
                    assessments.append(
                        {"case_id": row["case_id"], "label": label, "reason": reason}
                    )
                human_judge.save_reviews(args.run_id, reviewer, assessments)
                print("Review saved. Next: evals human-judge judge " + args.run_id)
            elif args.action == "judge":
                data = human_judge.judge(args.run_id, args.model)
                completed = sum(row["status"] == "completed" for row in data["cases"])
                print(f"Judged {completed} of {len(data['cases'])} answers.")
                for row in data["cases"]:
                    if row["status"] != "completed":
                        print(f"  {row['case_id']}: {row['error']}")
                if all(row["status"] == "completed" for row in data["cases"]):
                    print("Next: evals human-judge compare " + args.run_id)
                    return 0
                print("Judge incomplete. Retry this judge command before comparing.")
                return 1
            else:
                from ai_quality_evals_advanced.runtime import atomic_json

                comparison = redact(human_judge.compare(args.run_id))
                output_path = run_folder(args.run_id, ARTIFACTS) / (
                    f"comparison-{uuid.uuid4().hex}.json"
                )
                atomic_json(output_path, comparison)
                for row in comparison["cases"]:
                    print(f"\nCase: {row['case_id']}")
                    if row["answer_origin"] == "authored_control":
                        print("Origin: authored control (not a model-generated answer)")
                    print("Answer:", row["answer"] or "No answer generated")
                    print("Answer model:", row["answer_model"])
                    print("Your review:", row["human_label"], "—", row["human_reason"])
                    print("Judge:", row["judge_label"], "—", row["judge_reason"])
                    print("Judge model:", row["judge_model"])
                    print(
                        "Agreement:",
                        row["agreement"]
                        if row["agreement"] is not None
                        else "not compared",
                    )
                    if row["answer_origin"] == "authored_control":
                        detected = row["control_detected"]
                        print(
                            "Wrong answer detected:",
                            detected if detected is not None else "not checked",
                        )
                print(f"\nCompared: {comparison['compared']} of {comparison['total']}")
                print(
                    "Agreement:",
                    comparison["agreement"]
                    if comparison["agreement"] is not None
                    else "not available",
                )
                detected = comparison["controls_detected"]
                judged = comparison["controls_judged"]
                total = comparison["controls_total"]
                print(
                    f"Wrong-answer controls detected: {detected} of {judged} "
                    f"judged ({total} total)."
                )
                print(comparison["limitation"])
                if comparison["same_model_selection"]:
                    print(
                        "The same model selection was used for generation and judging."
                    )
                print("Saved comparison:", output_path)
            return 0
        if args.command == "compare":
            from ai_quality_evals_advanced.analysis import compare_saved

            print(json.dumps(redact(compare_saved(args.left, args.right)), indent=2))
            return 0
        if args.command == "regression":
            from ai_quality_evals_advanced.analysis import regression

            data = regression(args.variant)
            print(json.dumps(data, indent=2))
            return 0 if data["gate_pass"] else 1
        if args.command == "suite":
            from ai_quality_evals_advanced.experiments import run_experiment

            if args.models and args.name != "benchmark":
                raise ValueError("--models is only supported for benchmark")
            if args.provider:
                config = config.model_copy(update={"provider": args.provider})
            if config.provider == "openrouter":
                from ai_quality_evals.providers.openrouter import load_dotenv

                load_dotenv()
            data = run_experiment(
                args.name, args.mode, config, args.split, benchmark_models=args.models
            )
            display = dict(data)
            if "provenance" in display:
                display.pop("provenance")
                display["report_note"] = (
                    "Run details are available in the saved report."
                )
            print(json.dumps(redact(display), indent=2))
            return 0 if data["execution_status"] == "completed" else 1
        if args.command == "annotations":
            from ai_quality_evals_advanced.evaluation import (
                export_annotations,
                import_annotations,
            )

            if args.action == "export":
                export_annotations(args.path)
                print(
                    "Review exported; annotations remain empty until entered by people."
                )
            else:
                print(f"Validated {len(import_annotations(args.path))} annotations.")
            return 0
        if args.command == "phoenix":
            from ai_quality_evals_advanced.observability import serve

            return serve(config)
        if args.command == "doctor":
            data = doctor(config, args.offline)
            print(json.dumps(redact(data), indent=2))
            return (
                1
                if any(
                    c["status"] in {"missing", "unavailable"} for c in data["checks"]
                )
                else 0
            )
        if args.command == "validate":
            case = load_case()
            print(
                f"Configuration and case {case.case_id} valid; no inference attempted."
            )
            return 0
        if args.command == "run":
            result = execute(load_case(), args.mode, config)
        else:
            from ai_quality_evals_advanced.analysis import read_evidence

            data = read_evidence(args.run_id)
            if data.get("schema_version") == "2.0.0":
                print("RECORDED EVIDENCE - NO INFERENCE")
                print(json.dumps(redact(data), indent=2))
                return 0
            result = read_result(args.run_id)
        prefix = (
            "RECORDED EVIDENCE - NO INFERENCE (" + result.evidence_kind.upper() + ")"
            if args.command in {"report", "replay"}
            else result.evidence_kind.upper()
        )
        print(
            redact(
                f"{prefix}\nRun: {result.run_id}\nOutput: {result.output}\n"
                f"Verdict: {result.verdict}\nWhy: {result.explanation}\n"
                f"Trace: {result.trace_id}\nTelemetry: {result.telemetry['status']}"
            )
        )
        return (
            1 if args.command == "run" and result.execution_status != "completed" else 0
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(
            redact(f"Advanced setup/execution error: {type(exc).__name__}: {exc}"),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
