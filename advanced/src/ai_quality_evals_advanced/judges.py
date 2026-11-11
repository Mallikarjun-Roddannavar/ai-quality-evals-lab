"""Real DeepEval execution through a bounded approved-provider custom judge."""

import asyncio
import json
import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["DEEPEVAL_UPDATE_WARNING_OPT_OUT"] = "YES"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from deepeval.models import DeepEvalBaseLLM  # noqa: E402


class LocalJudge(DeepEvalBaseLLM):
    def __init__(self, inference):
        self.inference = inference
        super().__init__(model=inference.config.model)

    def load_model(self):
        return self.inference

    def get_model_name(self):
        return self.inference.config.model

    def generate(self, prompt, schema=None):
        if not isinstance(prompt, str):
            raise ValueError("Text-only local judge")
        reply = self.inference.generate(
            [{"role": "user", "content": prompt}],
            schema.model_json_schema() if schema is not None else None,
        )
        if reply.finish_reason != "stop":
            raise ValueError("Incomplete judge response")
        return schema.model_validate_json(reply.text) if schema else reply.text

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)


def faithfulness(inference, question: str, answer: str, context: list[str]) -> dict:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    metric = FaithfulnessMetric(
        model=LocalJudge(inference),
        async_mode=False,
        include_reason=True,
        threshold=0.8,
    )
    metric.measure(
        LLMTestCase(input=question, actual_output=answer, retrieval_context=context)
    )
    return {
        "framework": "DeepEval",
        "criterion": "faithfulness",
        "score": metric.score,
        "reason": metric.reason,
        "threshold": 0.8,
        "limitation": "Support assessment; not completeness or human calibration.",
    }


def python_judge(inference, question: str, answer: str, context: list[str]) -> dict:
    from pydantic import BaseModel, ConfigDict

    class Verdict(BaseModel):
        model_config = ConfigDict(extra="forbid")
        supported: bool
        complete: bool
        reason: str

    prompt = json.dumps(
        {
            "instruction": (
                "Judge only against supplied facts. "
                "Return supported, complete and reason."
            ),
            "facts": context,
            "question": question,
            "answer": answer,
        }
    )
    verdict = LocalJudge(inference).generate(prompt, Verdict)
    return {"criterion": "supported_and_complete", **verdict.model_dump()}


def preference_probe(
    inference, question: str, answers: list[str], facts: list[str]
) -> dict:
    from typing import Literal

    from pydantic import BaseModel

    class Choice(BaseModel):
        choice: Literal["A", "B", "tie"]
        reason: str

    if len(answers) != 2:
        raise ValueError("Two answers required")
    selections = []
    for order in ([0, 1], [1, 0]):
        choice = LocalJudge(inference).generate(
            json.dumps(
                {
                    "instruction": "Choose the better supported answer or tie.",
                    "question": question,
                    "facts": facts,
                    "A": answers[order[0]],
                    "B": answers[order[1]],
                }
            ),
            Choice,
        )
        selections.append(
            {
                "order": order,
                **choice.model_dump(),
                "selected_original": None
                if choice.choice == "tie"
                else order[choice.choice == "B"],
            }
        )
    return {
        "trials": selections,
        "consistent": selections[0]["selected_original"]
        == selections[1]["selected_original"],
        "limitation": "One swapped pair cannot estimate bias prevalence.",
    }
