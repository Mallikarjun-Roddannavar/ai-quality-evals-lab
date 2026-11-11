"""Small actual framework trials on matched DemoShop criteria."""

import asyncio
import math
import os
from contextlib import contextmanager

os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


@contextmanager
def local_cache(folder):
    keys = {
        "LOCALAPPDATA": str((folder / "sdk-cache").resolve()),
        "XDG_CACHE_HOME": str((folder / "sdk-cache").resolve()),
        "XDG_DATA_HOME": str((folder / "sdk-data").resolve()),
    }
    previous = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def retrieval_metrics(retrieved: list[str], relevant: list[str]) -> dict:
    if len(set(retrieved)) != len(retrieved) or not relevant:
        raise ValueError("Unique retrieved IDs and nonempty relevance labels required")
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import IDBasedContextRecall

    binary = [int(i in relevant) for i in retrieved]
    recall = len(set(retrieved) & set(relevant)) / len(set(relevant))
    reciprocal = next((1 / (i + 1) for i, v in enumerate(binary) if v), 0)
    sample = SingleTurnSample(
        retrieved_context_ids=retrieved, reference_context_ids=relevant
    )
    score = asyncio.run(IDBasedContextRecall().single_turn_ascore(sample))
    return {
        "python_recall": recall,
        "ragas_id_recall": score,
        "mrr": reciprocal,
        "precision": sum(binary) / len(binary) if binary else 0,
        "ndcg": sum(v / math.log2(i + 2) for i, v in enumerate(binary))
        / (
            sum(1 / math.log2(i + 2) for i in range(min(len(relevant), len(binary))))
            or 1
        ),
        "criterion": "document-ID retrieval; not answer faithfulness",
    }


def promptfoo_security(folder):
    import json
    import shutil
    import subprocess

    from ai_quality_evals_advanced.contracts import ROOT

    node = shutil.which("node")
    package = ROOT / "advanced/node_modules/promptfoo/package.json"
    if node is None or not package.exists():
        raise ValueError("Install Node and run npm ci --prefix advanced")
    metadata = json.loads(package.read_bytes())
    binary = metadata["bin"]
    entry = package.parent / (
        binary["promptfoo"] if isinstance(binary, dict) else binary
    )
    report = folder / "promptfoo.json"
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            s in k.upper()
            for s in ("API_KEY", "SECRET", "LANGSMITH", "LANGCHAIN", "OTEL_")
        )
    }
    env.update(
        PROMPTFOO_DISABLE_TELEMETRY="1",
        PROMPTFOO_DISABLE_UPDATE="1",
        PROMPTFOO_DISABLE_REMOTE_GENERATION="true",
        IS_TESTING="true",
        PROMPTFOO_CONFIG_DIR=str((folder / "promptfoo-state").resolve()),
    )
    command = [
        node,
        str(entry),
        "eval",
        "-c",
        str(ROOT / "advanced/security/promptfooconfig.yaml"),
        "--no-cache",
        "--no-progress-bar",
        "--max-concurrency",
        "1",
        "-o",
        str(report.resolve()),
    ]
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.returncode or not report.exists():
        raise RuntimeError("promptfoo local evaluation failed; no success fabricated")
    data = json.loads(report.read_bytes())
    return {
        "framework": "promptfoo",
        "version": metadata["version"],
        "stats": data["results"]["stats"],
        "report": "promptfoo.json",
        "criterion": "authored domain tool requests, not jailbreak prevalence",
    }


def inspect_benchmark(inference, samples, folder):
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["INSPECT_PY_LOGGER_FILE"] = str((folder / "inspect.log").resolve())
    os.environ["INSPECT_TRACE_FILE"] = str((folder / "inspect-trace.log").resolve())
    os.environ["INSPECT_EVAL_CTL_SERVER"] = "false"
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import GenerateConfig, Model, ModelAPI, ModelOutput, modelapi
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate

    @modelapi(name="demoshop")
    class ApprovedAPI(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            if tools:
                raise ValueError("This benchmark adapter is text-only")
            messages = [{"role": m.role, "content": m.text} for m in input]
            reply = inference.generate(messages)
            return ModelOutput.from_content(model=reply.model, content=reply.text)

    model = Model(
        ApprovedAPI(inference.config.model), config=GenerateConfig(max_tokens=128)
    )
    task = Task(
        dataset=[
            Sample(id=s["id"], input=s["request"], target=s["reference"])
            for s in samples
        ],
        solver=generate(),
        scorer=match(location="exact"),
    )
    with local_cache(folder):
        logs = eval(
            task,
            model=model,
            log_dir=str(folder),
            display="none",
            max_samples=1,
            max_tasks=1,
            max_retries=0,
        )
    return [
        {
            "status": log.status,
            "error": str(log.error.message) if log.error else None,
            "samples": [
                {
                    "id": s.id,
                    "output": s.output.completion,
                    "response_model": s.output.model,
                    "scores": {k: v.model_dump() for k, v in (s.scores or {}).items()},
                }
                for s in (log.samples or [])
            ],
        }
        for log in logs
    ]
