"""One-command browser entry point for the DemoShop evaluation starter lab."""

import argparse
import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

from ai_quality_evals import lab_evals as demos
from ai_quality_evals.lab_systems import ROOT
from ai_quality_evals.providers.openrouter import load_dotenv

QUICK_GUIDES = {
    "01": {
        "tests": "Whether an answer matches the expected price wording.",
        "flow": (
            "The model answers from shop facts; exact and normalized string "
            "checks compare it with `INR 800`."
        ),
        "status": (
            "PASS requires the exact expected string. Other wording goes to "
            "review, even if it may be factually right."
        ),
        "example": (
            "`INR 800` matches exactly; `The backpack costs 800 INR` may be "
            "right but goes to review."
        ),
    },
    "02": {
        "tests": (
            "Whether a deterministic rule can extract the backpack price "
            "and match the catalog."
        ),
        "flow": (
            "The model answers; a bounded parser reads a supported amount "
            "and compares it with the fixture price of INR 800."
        ),
        "status": (
            "PASS means the recognized price matches; FAIL means it differs; "
            "NEEDS REVIEW means the wording is unsupported or ambiguous."
        ),
        "example": "`Backpack costs 800 INR` passes; `INR 900` fails.",
    },
    "03": {
        "tests": "How a person labels one answer and explains the judgment.",
        "flow": (
            "Read the answer and source facts, then enter your label and "
            "reason in the assessment form."
        ),
        "status": (
            "The human label is saved separately. One assessment does not "
            "measure reviewer agreement."
        ),
        "example": (
            "Label `INR 100` correct because the Notebook fixture price is 100."
        ),
    },
    "04": {
        "tests": (
            "Whether a model judge recognizes an answer that admits a policy "
            "is missing."
        ),
        "flow": (
            "One model answers the shipping question; a judge compares it "
            "with the supplied policies and returns a verdict and reason."
        ),
        "status": (
            "Read the judge's verdict and reason. NEEDS REVIEW remains because "
            "this judge has not been calibrated against human labels."
        ),
        "example": (
            "With no international-shipping policy, `I don't have that "
            "information` is supported; `We don't ship internationally` invents a rule."
        ),
    },
    "05": {
        "tests": "How two models answer the same small set of shop questions.",
        "flow": (
            "Each model receives the same three questions; supported catalog "
            "facts get narrow checks and usage is shown."
        ),
        "status": (
            "Compare the rows. Shipping remains ungraded, and this small set "
            "does not establish a winning model."
        ),
        "example": (
            "Both models answer `How much is the Notebook?`; compare each "
            "answer with the fixture's INR 100 price."
        ),
    },
    "06": {
        "tests": (
            "Whether a second turn resolves 'it' from the earlier backpack context."
        ),
        "flow": (
            "The host sends turn one, then turn two with prior user and "
            "assistant messages; a catalog check looks for INR 800."
        ),
        "status": (
            "PASS/FAIL applies to a recognized price; NEEDS REVIEW means the "
            "answer wording could not be safely parsed."
        ),
        "example": (
            "Turn 1 mentions the Backpack; turn 2 asks `How much does it "
            "cost?`; the expected price is INR 800."
        ),
    },
    "07": {
        "tests": (
            "Whether the cancellation policy is retrieved and referenced in an answer."
        ),
        "flow": (
            "Keyword search selects a policy passage; the model answers with "
            "that passage in context; retrieval and citation are checked separately."
        ),
        "status": (
            "FAIL means retrieval missed the expected document. NEEDS REVIEW "
            "means it found it, while answer support still needs human inspection. "
            "A citation marker alone is not proof."
        ),
        "example": (
            "Retrieve `cancellation`: `Cancellation is allowed only before "
            "shipment`; an answer may cite `[cancellation]`."
        ),
    },
    "08": {
        "tests": (
            "Whether the model selects the product lookup and reports its "
            "result correctly."
        ),
        "flow": (
            "The model returns a native tool call; the host validates and runs "
            "the lookup, returns its result, then asks for a final answer."
        ),
        "status": (
            "PASS means the tool returned the expected fact and the final "
            "answer passed its check. FAIL means the selection or fact was wrong; "
            "unclear final wording needs review."
        ),
        "example": (
            "The model requests `get_product(SKU-NOTEBOOK)`; the host returns "
            "price 100, which grounds the final answer."
        ),
    },
    "09": {
        "tests": (
            "Whether product lookup works through MCP and bad arguments are rejected."
        ),
        "flow": (
            "An MCP client starts the local stdio server, discovers its tools, "
            "makes a valid lookup, then sends malformed arguments."
        ),
        "status": (
            "PASS means the valid result matches the catalog and malformed "
            "arguments are rejected; otherwise the protocol check fails."
        ),
        "example": (
            "`get_product(product_id=SKU-NOTEBOOK)` returns the Notebook; "
            "`get_product(wrong_argument=...)` is rejected."
        ),
    },
    "10": {
        "tests": (
            "Whether the application loads the cancellation procedure for a "
            "matching request, but not for a price question."
        ),
        "flow": (
            "A host keyword rule selects `SKILL.md`, injects it into the "
            "prompt, and the host executes validated sandbox actions. This "
            "is application routing, not native coding-assistant skill discovery."
        ),
        "status": (
            "PASS requires the order to be cancelled and the price question "
            "not to trigger the procedure. The run also displays whether the "
            "procedure entered the prompt."
        ),
        "example": (
            "`Cancel DS-1001` loads the procedure; `What is the Notebook "
            "price?` loads none."
        ),
    },
    "11": {
        "tests": (
            "Whether a single agent can inspect and cancel an eligible owned order."
        ),
        "flow": (
            "The model returns JSON actions; the host validates and dispatches "
            "`get_order` or `cancel_order`, then returns tool results."
        ),
        "status": (
            "PASS requires the sandbox order to end as `cancelled`; a model "
            "claim alone is not enough."
        ),
        "example": (
            "Model requests `get_order(DS-1001)`, sees `awaiting shipment`, "
            "then requests `cancel_order(DS-1001)`."
        ),
    },
    "12": {
        "tests": (
            "How a single role compares with a policy reviewer handing off "
            "to an order worker."
        ),
        "flow": (
            "The single-agent path decides both actions. In the two-role path, "
            "the host inspects first, a reviewer returns a verdict, then a worker acts."
        ),
        "status": (
            "PASS means both fresh orders ended as cancelled. Compare calls "
            "and handoff too; one run cannot show delegation is better."
        ),
        "example": (
            "Reviewer returns `allowed`; the Order worker gets that verdict "
            "and requests cancellation."
        ),
    },
    "13": {
        "tests": "Whether one customer can read or change another customer's order.",
        "flow": (
            "The host tool tries cross-customer read and cancellation, checks "
            "legitimate access, and compares shop state before and after."
        ),
        "status": (
            "PASS means unauthorized actions were denied, the other order "
            "stayed unchanged, and the caller's own order remained accessible."
        ),
        "example": (
            "Caller `C001` tries to read `DS-1003`, owned by `C002`; access "
            "is denied and state stays unchanged."
        ),
    },
    "14": {
        "tests": (
            "How a pass rate and Wilson interval describe uncertainty in a small set."
        ),
        "flow": (
            "A deterministic catalog check scores five authored answers; "
            "code calculates the pass rate and interval."
        ),
        "status": (
            "PASS means the calculation ran. These constructed examples are "
            "not a representative estimate of model accuracy."
        ),
        "example": "Three passes out of five is 60%, with a wide interval.",
    },
    "15": {
        "tests": "What usage the provider reports for one live answer call.",
        "flow": (
            "The model answers once; the lab reads token, latency, and cost "
            "fields from the provider response when available."
        ),
        "status": (
            "PASS means token measurements were present; NEEDS REVIEW means "
            "one or more required measurements were unavailable."
        ),
        "example": (
            "One answer call reports input/output tokens and provider latency; "
            "missing usage stays unknown."
        ),
    },
    "16": {
        "tests": (
            "Whether a local regression check catches a deliberately wrong "
            "notebook price."
        ),
        "flow": (
            "The same check runs on a valid baseline and a controlled broken "
            "case, then compares the outcomes."
        ),
        "status": (
            "PASS means the baseline passed and the seeded error was caught. "
            "This is local, not a configured remote CI pipeline."
        ),
        "example": "Notebook INR 100 passes; the seeded INR 150 answer is caught.",
    },
}


def code_excerpt(topic_id: str) -> str:
    source = inspect.getsource(demos.run_demo)
    candidates = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.If) and any(
            isinstance(value, ast.Constant) and value.value == topic_id
            for value in ast.walk(node.test)
        ):
            candidates.append(node)
    if not candidates:
        return source
    node = max(candidates, key=lambda value: value.lineno)
    return "\n".join(source.splitlines()[node.lineno - 1 : node.body[-1].end_lineno])


def show_output(st, result: demos.DemoResult) -> None:
    output = result.output
    topic = result.topic_id
    st.subheader("What happened")
    if result.status == "execution_error":
        st.write(output)
    elif isinstance(output, str):
        st.write(output)
    elif topic == "04":
        st.write(output["answer"])
        st.write("Judge:", output["judge"]["verdict"], "—", output["judge"]["reason"])
        if result.provenance.get("judge_is_subject_model"):
            st.caption(
                "The same model generated and judged this answer. This is "
                "not independent validation."
            )
    elif topic == "05":
        st.dataframe(
            [
                {
                    "Model": row["model"],
                    "Question": row["question"],
                    "Answer": row["answer"],
                    "Catalog check": row["check"]["status"],
                }
                for row in output
            ],
            hide_index=True,
            width="stretch",
        )
    elif topic == "06":
        st.write("Turn 1:", output["turn_1"])
        st.write("Turn 2:", output["turn_2"])
    elif topic == "07":
        st.write(output["answer"])
        st.caption("Retrieved document: " + output["retrieved"]["document_id"])
        st.write(output["retrieved"]["text"])
    elif topic == "08":
        st.write("Model selected:", output["model_selection"]["tool"])
        st.write("Actual tool result:", output["actual_tool_result"])
        st.write("Final answer:", output.get("final_answer"))
        st.caption(
            "Native Ollama tool_calls response followed by a tool-role result."
            if "native_message" in output
            else "Historical structured-action example; this run predates native tools."
        )
    elif topic == "09":
        st.write("Connected to the local MCP server over stdio.")
        st.write("Discovered tools:", ", ".join(output["tools"]))
        st.write(
            "Product lookup returned the catalog record; malformed "
            "arguments were rejected."
        )
    elif topic == "10":
        st.write(
            "Loaded file:",
            output["matching_request"].get("path", "Historical inline procedure"),
        )
        st.write("Cancellation request →", output["matching_request"]["name"])
        st.write("Price question → no procedure loaded")
        for step in output["matching_request"]["instructions"]:
            st.write("•", step)
        if "execution" in output:
            st.write(
                "Actual final order status:",
                output["execution"]["final_order"]["status"],
            )
            with st.expander("Read the loaded SKILL.md instructions"):
                st.markdown(output["matching_request"]["body"])
    elif topic == "11":
        st.write("Final order status:", output["final_order"]["status"])
        st.write("Last actual outcome:", output["output"])
    elif topic == "12":
        st.table(
            [
                {
                    "Variant": name,
                    "Final status": row["final_order"]["status"],
                    "Model calls": row["usage"]["model_calls"],
                    "Tool calls": row["usage"]["tool_calls"],
                }
                for name, row in output.items()
            ]
        )
        st.write("Role handoff:", output["two_roles"]["handoff"])
    elif topic == "13":
        st.write(output["cross_customer_read"]["error"])
        st.write(output["cross_customer_cancel"]["error"])
        st.write(
            "Your own order remains accessible. Order state unchanged:",
            output["state_unchanged"],
        )
    elif topic == "14":
        lower, upper = output["wilson_95"]
        st.write(
            f"{output['passes']} of {output['scored']} authored examples pass: "
            f"{output['pass_rate']:.0%}."
        )
        st.write(f"Illustrative Wilson 95% interval: {lower:.1%} to {upper:.1%}.")
    elif topic == "15":
        st.write(output["answer"])
    elif topic == "16":
        st.write("Baseline:", "PASS" if output["baseline"]["passed"] else "FAIL")
        st.write(
            "Deliberately wrong price:",
            "CAUGHT" if not output["controlled_regression"]["passed"] else "MISSED",
        )
    else:
        st.write(output)


def render() -> None:
    load_dotenv()
    import streamlit as st

    st.set_page_config(
        page_title="DemoShop Evaluation Lab", page_icon="🧪", layout="wide"
    )
    st.title("DemoShop Evaluation Lab")
    st.write("One small shop. Sixteen simple ways to evaluate it.")
    st.caption("Starter demonstrations first. Full module validation comes later.")
    labels = {row[0]: f"{row[0]} · {row[1]}" for row in demos.TOPICS}
    topic_id = st.selectbox("Choose a topic", list(labels), format_func=labels.get)
    topic = next(row for row in demos.TOPICS if row[0] == topic_id)
    st.write(topic[2])
    st.caption("DemoShop mode: " + topic[3])
    guide = QUICK_GUIDES[topic_id]
    with st.expander("Quick learning notes: what this tests and how it works"):
        st.markdown("**What this tests:** " + guide["tests"])
        st.markdown("**Internal flow:** " + guide["flow"])
        st.markdown("**Reading the result:** " + guide["status"])
        st.markdown("**Small example:** " + guide["example"])
    with st.expander("Advanced settings"):
        provider = st.selectbox(
            "Inference provider",
            ["ollama", "openrouter"],
            format_func=lambda value: {
                "ollama": "Ollama (local)",
                "openrouter": "OpenRouter (hosted)",
            }[value],
            index=0 if os.getenv("AI_EVAL_PROVIDER", "ollama") == "ollama" else 1,
        )
        openrouter_selected = provider == "openrouter"
        model_default = (
            os.getenv("OPENROUTER_MODEL", "").strip() or "openai/gpt-4o-mini"
            if openrouter_selected
            else os.getenv("OLLAMA_MODEL", "").strip() or "gemma4:e2b"
        )
        model = st.text_input(
            "Assistant model", value=model_default, key=f"assistant_model_{provider}"
        )
        secondary_default = (
            os.getenv("OPENROUTER_SECONDARY_MODEL", "").strip()
            or "google/gemini-2.5-flash"
            if openrouter_selected
            else "gemma2:2b"
        )
        judge_default = (
            os.getenv("OPENROUTER_JUDGE_MODEL", "").strip() or model_default
            if openrouter_selected
            else "gemma4:e2b"
        )
        secondary = st.text_input(
            "Second comparison model",
            value=secondary_default,
            key=f"secondary_model_{provider}",
        )
        judge = st.text_input(
            "Judge model", value=judge_default, key=f"judge_model_{provider}"
        )
        if openrouter_selected:
            if os.getenv("OPENROUTER_API_KEY", "").strip():
                st.caption(
                    "OpenRouter key detected in OPENROUTER_API_KEY. "
                    "The key is read from the environment and never shown or saved."
                )
            else:
                st.warning(
                    "Set OPENROUTER_API_KEY in PowerShell, then restart the lab. "
                    "The API key is never entered in this page or stored in reports."
                )
            st.caption(
                "Use an OpenRouter model ID available to your account. "
                "Structured-output and tool topics require model support for "
                "those parameters. Model calls may incur provider charges."
            )
        else:
            st.caption(
                "Use installed Ollama tags. Models are never downloaded automatically."
            )
    runs = demos.saved_runs(topic_id)
    first, second = st.columns(2)
    run_clicked = first.button("Run demo", type="primary", width="stretch")
    replay_clicked = second.button(
        "Replay latest saved run", disabled=not runs, width="stretch"
    )
    key = "lab_result_" + topic_id
    if run_clicked:
        with st.spinner("Running this example…"):
            result = demos.run_demo(
                topic_id,
                model=model,
                secondary_model=secondary,
                judge_model=judge,
                provider=provider,
            )
            try:
                path = demos.save_result(result)
                st.session_state[key] = result
                st.session_state[key + "_path"] = str(path)
            except OSError as exc:
                st.error(f"Could not save evidence: {exc}")
    elif replay_clicked:
        try:
            st.session_state[key] = demos.replay(runs[0])
            st.session_state[key + "_path"] = str(runs[0])
        except (OSError, ValueError) as exc:
            st.error(f"Could not replay saved evidence: {exc}")
    result = st.session_state.get(key)
    if result is None:
        st.info(
            "Choose a topic and run its small example. No model call "
            "happens until you click Run demo."
        )
        return
    evidence_label = {
        "live": "LIVE MODEL OUTPUT",
        "controlled": "CONTROLLED LOCAL EXAMPLE",
        "replay": "RECORDED REPLAY — NO NEW MODEL CALL",
    }
    st.caption(evidence_label[result.evidence_kind])
    facts_column, output_column = st.columns([1, 2])
    with facts_column:
        st.subheader("Shop facts")
        st.table(
            [
                {
                    "Product": p["name"],
                    "Price (INR)": p["price_inr"],
                    "Stock": p["stock"],
                }
                for p in result.source_facts["products"]
            ]
        )
        st.caption(
            "Returns: 14 days after delivery. Cancellation: before "
            "shipment only. Own orders only."
        )
        st.write("Expected:", result.expected)
    with output_column:
        show_output(st, result)
        display_status = result.status.replace("_", " ").upper()
        method = (
            st.success
            if result.status == "pass"
            else st.error
            if result.status in {"fail", "execution_error"}
            else st.warning
        )
        method(display_status + " — " + result.explanation)
        st.caption(result.limitation)
    if topic_id in {"03", "04"} and result.status != "execution_error":
        st.subheader("Your assessment")
        with st.form("human_" + result.run_id):
            reviewer = st.text_input("Your name")
            label = st.selectbox(
                "Your label", ["correct", "incorrect", "uncertain"], index=None
            )
            reason = st.text_area("Why?")
            submitted = st.form_submit_button("Save my assessment")
        if submitted:
            try:
                demos.save_human_review(result, reviewer, label, reason)
                st.success(
                    "Your assessment was saved separately from the original "
                    "model evidence."
                )
                if topic_id == "04":
                    st.write(
                        "Judge label:",
                        result.output["judge"]["verdict"],
                        "Your label:",
                        label,
                    )
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    if result.usage["model_calls"]:
        cost = result.usage["cost"]
        cost_note = (
            f"OpenRouter-reported cost ${cost:.8f} USD; local energy unmeasured."
            if cost is not None
            else result.usage["cost_note"]
        )
        st.caption(
            f"Observed: {result.usage['model_calls']} model calls · "
            f"{result.usage['completion_tokens']} output tokens · "
            f"{result.usage['latency_ms'] / 1000:.1f}s provider time. " + cost_note
        )
    with st.expander("Inspect code, traces and saved evidence"):
        st.write("Saved evidence:", st.session_state.get(key + "_path", ""))
        st.code(code_excerpt(topic_id), language="python")
        st.json(result.model_dump(mode="json"))
        st.download_button(
            "Download result JSON",
            result.model_dump_json(indent=2),
            file_name=f"demoshop-{topic_id}-{result.run_id}.json",
            mime="application/json",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Run the offline CI-ready price check"
    )
    parser.add_argument(
        "--broken",
        action="store_true",
        help="Use a clearly labeled controlled regression",
    )
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if args.check:
        result = demos.regression_check(broken=args.broken)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true",
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    render()
