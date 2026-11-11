# Advanced tools and trace matrix

This is a map of the **Advanced** learning workflows: what each tool does,
where it is used, and what to look for in Phoenix or the saved report. The
examples use fictional DemoShop data and demonstrate narrow checks.

## The tools at a glance

There are **14 entries in this learning map**. This is a practical grouping,
not a count of every Python package or command-line utility installed.

| # | Tool | Role | Used in |
|---|---|---|---|
| 1 | Python | Builds the workflows and implements explicit checks. | All Advanced topics. |
| 2 | pytest | Runs automated tests, including offline model doubles. | Project verification, not the live evaluation itself. |
| 3 | Pydantic | Validates configuration, datasets and structured model results. | Shared contracts, human review, judges and agent handoffs. |
| 4 | DeepEval | Runs a model-based faithfulness metric. | `judges`: compares the framework check with a separate Python judge. |
| 5 | Ragas | Calculates document-ID retrieval recall. | `retrieval` offline; this does not judge answer support. |
| 6 | Inspect AI | Runs model benchmark samples and an exact-match scorer. | `benchmark`, alongside a separate Python price check. |
| 7 | LangChain | Builds a chat or retrieval application variant. | `chat`, `rag`; it does not score the answers. |
| 8 | LangGraph | Builds a graph-based agent workflow. | `agents`, alongside the Python agent variant. |
| 9 | promptfoo | Runs assertions against local security scenarios. | The offline `promptfoo` experiment. |
| 10 | MCP Python SDK | Connects a real local client and server to discover and call tools. | The offline `mcp` experiment. |
| 11 | OpenTelemetry | Creates and exports spans from the application. | Instrumented Advanced runs. |
| 12 | Phoenix | Receives and displays local traces. | Instrumented runs when Phoenix is running and export succeeds. |
| 13 | OpenRouter | Routes requests to a selected hosted model. | Live model workflows when configured as the provider. |
| 14 | Ollama | Runs configured local models. | Alternative provider; semantic retrieval also requires an installed local embedding model. |

`uv`, Node.js and npm support installation or execution; they are not
evaluation methods. `get_product`, `get_order` and `cancel_order` are **shop
tools called by an assistant**, a different meaning of “tool” from the
frameworks in the table.

## Topic → framework → trace

Use the commands in the [Advanced README](README.md). An `experiment.*` trace
generally contains the spans listed below. A span exists only when that part of
the workflow is actually reached; an early failure can stop later steps.

| Topic | Main tools or technique | What to look for in Phoenix or reports |
|---|---|---|
| 01 Foundations | Python exact and normalized matching; authored cases | `evaluation.case.*` shows a case and checks; `evaluation.summary` shows counts. No model call. |
| 02 Deterministic | Python rules and controlled regression cases | `suite regression --mode offline` shows `evaluation.regression` and its check results. No model call. |
| 03 Human review | Human labels, Python and OpenRouter | `human_judge.generate.case` and `model.generate`; the interactive review and comparison are saved locally, not exported as their own traces. |
| 04 Model judges | Python judge, DeepEval, Pydantic and model provider | `judge.python`, `judge.deepeval`, `model.generate`, and `evaluation.case.*`; inspect errors as well as verdicts. |
| 05 Benchmarking | Inspect AI and Python price checks | `model.generate`, `evaluation.benchmark.case`, and each model's Inspect logs and saved report. |
| 06 Chat | Python and LangChain chat variants | `chat.python.turn.*`, `chat.langchain.turn.*`, and `model.generate` show message history and responses. |
| 07 RAG | Python lexical retrieval; LangChain wrapper; Ragas in the separate retrieval experiment | `retrieval` shows the question and documents; `model.generate` shows the augmented prompt and answer; `evaluation.variant` shows the citation check. |
| 08 Tool calling | Model provider and host Python tools | `model.generate` shows requested tool calls; `tool.get_product` shows host execution and outcome. |
| 09 MCP | MCP SDK and Python domain tools; a model in `mcp-selection` | The offline `mcp` experiment checks protocol calls. Live `mcp-selection` compares actual and misleading descriptions using `mcp.selection.*`, `model.generate`, and `mcp.call.get_order`. |
| 10 Skills | Python router and file-backed `SKILL.md` | `evaluation.case.*` shows the selection and expected skill. The offline selection demo makes no model call. |
| 11 Single agent | Python or LangGraph, provider and host tools | `agent.phase.*`, `agent.order_worker`, `model.generate`, and `tool.*` show the sequence and state. |
| 12 Multi-agent | Python or LangGraph with reviewer and worker roles | `agent.policy_reviewer`, `agent.order_worker`, `model.generate`, `tool.*`, and phase snapshots show the handoff. |
| 13 Security | Python authorization checks; promptfoo in the offline security suite | `tool.*`, agent phases, and `evaluation.case.*`; `suite promptfoo` saves its detailed framework report. Expected denials are not provider errors. |
| 14 Statistics | Python reliability and uncertainty calculations | `evaluation.result` and available `evaluation.summary` snapshots; no model call. |
| 15 Operations | Python timing and token summaries; model provider | `model.generate` spans and `evaluation.case.*` show attempts, usage and results; unreported cost is unknown. |
| 16 Regression | Python controlled baseline and defects | `suite regression --mode offline` shows `evaluation.regression`; the standalone `regression --variant` command does not export its own trace. |

## How to read a run

1. Start Phoenix with `uv run --extra advanced evals phoenix`, then run an
   experiment in another terminal. Open the experiment trace in Phoenix.
2. Expand `model.generate` to see the request and response. If the call failed,
   inspect its error; no response is invented. Tool calls have their own spans.
3. Open `evaluation.case.*` or `evaluation.result` for the checks. These are
   **snapshots** recorded after evaluation; `agent.phase.*` records state at
   the start of a phase. Their span durations do not time the described work.
4. Find the local files under `reports/advanced/RUN_ID/`: `experiment.json`
   contains results, `traces.jsonl` contains local spans, and `events.jsonl`
   records instrumented calls when events were produced. Human-judge workflows
   additionally save their review and comparison files there.

`execution_status: completed` means execution completed, not that the answer
is correct. Inspect the stated check and review any provider or grading errors.
Local trace files can exist even when Phoenix export fails; inspect the run's
`telemetry.status` for export status. Model inputs and outputs are captured for
these synthetic examples; do not use real customer data in the learning runs.
