"""Controlled decisions test actual Python/LangGraph transitions and domain state."""

import json

import pytest

from ai_quality_evals_advanced.agents import agent_run
from ai_quality_evals_advanced.contracts import Reply
from ai_quality_evals_advanced.security_target import target
from ai_quality_evals_advanced.telemetry import Telemetry


class ControlInference:
    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.attempts = []
        self.acted = False

    def generate(self, messages, schema=None, tools=None):
        self.attempts.append({"status": "completed", "elapsed_ms": 0})
        if schema:
            return Reply(
                model="authored-control",
                text=json.dumps(
                    {
                        "order_id": "DS-1001",
                        "recommendation": "Cancel awaiting shipment",
                        "evidence": ["awaiting shipment"],
                        "uncertainty": "none in fixture",
                    }
                ),
            )
        if not self.acted:
            self.acted = True
            return Reply(
                model="authored-control",
                text="",
                tool_calls=[
                    {
                        "function": {
                            "name": "cancel_order",
                            "arguments": {"order_id": "DS-1001"},
                        }
                    }
                ],
            )
        return Reply(model="authored-control", text="Cancelled")


@pytest.mark.parametrize("variant", ["python", "langgraph"])
@pytest.mark.parametrize("multi", [False, True])
def test_agents_compare_actual_state(tmp_path, variant, multi):
    telemetry = Telemetry(tmp_path, "control", None)
    try:
        result = agent_run(
            ControlInference(telemetry),
            tmp_path,
            "Cancel DS-1001",
            variant=variant,
            multi=multi,
        )
        assert result["final_state"]["DS-1001"]["status"] == "cancelled"
        assert result["tool_calls"] == 3
        assert result["model_calls"] == (3 if multi else 2)
        assert bool(result["handoff"]) == multi
        assert result["phases"][-2:] == ["verify", "respond"]
        assert result["error"] is None
    finally:
        telemetry.close()


@pytest.mark.parametrize("question", ["Cancel DS-1003", "Cancel my order"])
def test_denial_clarification_cannot_mutate(tmp_path, question):
    telemetry = Telemetry(tmp_path, "control", None)
    try:
        inference = ControlInference(telemetry)
        result = agent_run(
            inference, tmp_path, question, variant="langgraph", multi=True
        )
        assert result["model_calls"] == 0
        assert result["final_state"]["DS-1003"]["status"] == "delivered"
        assert result["output"]
    finally:
        telemetry.close()


def test_promptfoo_target_is_actual_domain(tmp_path):
    database = tmp_path / "security.sqlite"
    assert target({"tool": "cancel_order", "order_id": "DS-1001"}, database) == {
        "status": "allowed",
        "changed": True,
    }
    assert target({"tool": "cancel_order", "order_id": "DS-1003"}, database) == {
        "status": "denied",
        "changed": False,
    }
