import json

from ai_quality_evals_advanced.contracts import Reply
from ai_quality_evals_advanced.mcp_selection import provider_error, run_selection
from ai_quality_evals_advanced.providers import ProviderError
from ai_quality_evals_advanced.telemetry import Telemetry


def test_provider_error_inside_mcp_exception_group():
    error = ProviderError("transport_error")
    assert provider_error(ExceptionGroup("mcp", [error])) is error


def test_provider_failures_remain_errors(tmp_path):
    telemetry = Telemetry(tmp_path, "failure", None)

    class FailingModel:
        def __init__(self):
            self.telemetry = telemetry

        def generate(self, messages, tools):
            raise ProviderError("transport_error")

    try:
        result = run_selection(FailingModel(), tmp_path, "failure")
    finally:
        telemetry.close()
    assert [row["error_detail"] for row in result["cases"]] == [
        "transport_error",
        "transport_error",
    ]


def test_real_mcp_selection_and_description_change(tmp_path):
    telemetry = Telemetry(tmp_path, "control", None, capture_content=True)

    class ModelDouble:
        def __init__(self):
            self.telemetry = telemetry
            self.descriptions = []

        def generate(self, messages, tools):
            self.descriptions.append(
                {t["function"]["name"]: t["function"]["description"] for t in tools}
            )
            name = "get_order" if len(self.descriptions) == 1 else "cancel_order"
            return Reply(
                model="control",
                text="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps({"order_id": "DS-1001"}),
                        },
                    }
                ],
            )

    model = ModelDouble()
    try:
        result = run_selection(model, tmp_path, "control")
    finally:
        telemetry.close()
    normal, misleading = result["cases"]
    assert normal["server_descriptions_match_expected"] is True
    assert normal["presented_descriptions_match_expected"] is True
    assert misleading["presented_descriptions_match_expected"] is False
    assert normal["selection_correct"] is True
    assert normal["tool_outcome"]["is_error"] is False
    assert normal["tool_outcome"]["result"]["order_id"] == "DS-1001"
    assert normal["tool_result_matches_expected"] is True
    assert misleading["selected_tool"] == "cancel_order"
    assert misleading["selection_correct"] is False
    assert misleading["tool_outcome"] is None
    assert misleading["tool_result_matches_expected"] is None
    assert model.descriptions[0]["get_order"] != model.descriptions[1]["get_order"]
    traces = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {s["name"] for s in traces} >= {
        "mcp.selection.normal",
        "mcp.selection.misleading",
        "mcp.call.get_order",
    }
