# Advanced DemoShop

Evaluation, observability and security for the same small shop. Python library
and CLI; no custom browser app. Intro remains the beginner experience.

New to the code? Use the [code and libraries guide](CODE_GUIDE.md) for a
high-level map of the execution flow, files and dependencies.
For a topic-by-topic map of frameworks, model calls and Phoenix spans, see the
[tools and trace matrix](TRACE_MATRIX.md).

## Quick reminder: what is here?

The same fictional shop is used for small evaluation experiments. Start with a
question, decide what a good result looks like, run the case, and inspect both
the result and any failures. You do not need every tool for every evaluation.

| Tool | Why it is here | DemoShop example |
|---|---|---|
| Python and pytest | Write and check explicit rules and expected outcomes. | Check that the owned order actually became cancelled. |
| DeepEval | Explore model-based judging with a faithfulness metric. | Check whether an answer's claims are supported by the supplied facts. |
| Ragas | Explore retrieval metrics. | Check whether the relevant policy document was retrieved. |
| Inspect AI | Run benchmark tasks, solvers and scorers. | Ask the same product questions across models and compare scores. |
| LangChain | Build application variants to evaluate. | Compare a framework chat or RAG pipeline with a Python version. |
| LangGraph | Build agent workflows to evaluate. | Compare single-agent and multi-agent cancellation workflows. |
| promptfoo | Run local security assertions. | Check domain permission boundaries on authored requests. |
| MCP SDK | Run and inspect an actual local tool protocol. | Discover and call an order tool through MCP. |
| OpenTelemetry and Phoenix | Capture and inspect execution traces. | Follow model, tool and grader steps in a run. |

LangChain and LangGraph build applications; they do not grade correctness.
Phoenix shows what happened; a trace alone does not prove an answer is correct.
DeepEval and the Python judge use different criteria, so their scores are not
directly interchangeable. Ragas document-ID recall does not check whether the
generated answer is supported.

## Evaluation types at a glance

| What is evaluated | Example in this project |
|---|---|
| Answer correctness and deterministic rules | Match a price or check the actual order state. |
| Human review and model judging | Compare human labels with a Python judge; explore DeepEval separately. Broader judge calibration remains pending. |
| Model comparison and benchmarking | Run matched product questions and compare model results using Inspect and Python checks. |
| Conversation and RAG | Check a reference across turns; measure retrieved document IDs, ranking and citations separately. |
| Tools, MCP and skills | Check actual calls and arguments, protocol behavior, and skill selection. |
| Single and multi-agent behavior | Compare state outcomes, handoffs and calls from fresh starting states. |
| Security | Check ownership and permissions alongside legitimate operations. The local promptfoo suite is not a general jailbreak benchmark. |
| Reliability and operations | Examine statistical uncertainty, latency, calls and available token usage. |
| Regression | Detect controlled price, policy, retrieval and authorization defects. |

These can be combined. For example, compare two agent variants on the same
offline security cases and use Python to check the final state. The project has
matched comparisons and local regression checks; it does not implement a live
user-traffic A/B experiment, shadow deployment or canary rollout.

## Suggested learning path

1. Start with a small offline case and a Python check you understand.
2. Read the actual output and decide whether the check captured what matters.
3. Try a model judge or framework comparison and inspect where it disagrees.
4. Add a regression or security case for a failure you found.

For worked examples and limitations, see [LEARNING.md](LEARNING.md). The topic
table below maps these ideas to the available experiments.

## Start locally

From the repository root:

```powershell
uv sync --locked --extra dev --extra advanced
uv run --extra advanced evals doctor --offline
uv run --extra advanced evals validate
uv run --extra advanced evals phoenix
```

Phoenix runs as a native Python process at http://127.0.0.1:6006. No Docker.
Keep that terminal open. Its SQLite database stays under ignored
`reports/advanced/phoenix/`. On this Windows/Python 3.14 environment, the pinned
Phoenix dependency required a native build using existing Rust/MSVC tools.
Installation on a fresh machine may require those build tools.

In Phoenix, open the latest experiment trace and select a child span to see its
attributes. Foundations shows each case ID, exact/normalized check and authored
label, followed by summary counts. Skills, retrieval, statistics, regression and
MCP also include compact check details. Case and summary spans are recorded after
the checks complete; their durations measure recording, not the original work.
Live runs record model input and output in model-call spans and the local event
log. Evaluation snapshots are separate summaries; their coverage varies by
topic. Keep the captured content limited to fictional DemoShop examples.

In another terminal:

```powershell
uv run --extra advanced evals run --example price --mode offline
uv run --extra advanced evals run --example price --mode live
uv run --extra advanced evals report latest
uv run --extra advanced evals replay latest
```

Offline uses a clearly labeled authored answer. Live explicitly calls the
installed Ollama model `gemma4:e2b`; missing models produce an error, never a
download or simulated fallback. Report/replay only read recorded evidence.
Configuration is in [configs/local.json](configs/local.json).

## Experiments across the 16 topics

### Commands from our walkthrough

Run these from the repository root. The first exercise works offline. The
human-review and benchmark exercises use OpenRouter; add your key to the
root `.env` before running them. Start Phoenix in a separate terminal if you
want to inspect traces; it is not the evaluator.

**1. Foundations — exact and normalized checks**

```powershell
uv run --extra advanced evals suite foundations --mode offline
```

Python compares prepared answers and expected labels using exact and normalized
matching. Inspect individual cases and false passes or failures; these are
authored examples, not live model answers.

**2. Human review and LLM judge — compare assessments**

```powershell
uv run --extra advanced evals human-judge generate --model openrouter/free --limit 5
uv run --extra advanced evals human-judge review RUN_ID
uv run --extra advanced evals human-judge judge RUN_ID --model openrouter/free
uv run --extra advanced evals human-judge compare RUN_ID
```

Replace `RUN_ID` with the ID printed by `generate` in each later command. Review
each saved answer and provide a label and a nonempty reason. The judge evaluates
those same answers; the comparison also checks whether it detects one authored
wrong-price control. This workflow uses Python and OpenRouter; it does not use
DeepEval. The separate `judges` suite below demonstrates DeepEval. Retry the
judge command if it reports incomplete results, before comparing.

**3. Benchmark — compare two models**

```powershell
uv run --extra advanced evals suite benchmark --mode live --provider openrouter --models MODEL_A MODEL_B
```

Replace `MODEL_A` and `MODEL_B` with two explicit OpenRouter model IDs. Inspect
AI checks the requested answer format and Python checks the price on the same
three product questions. One model completed all three cases in our walkthrough;
the second returned an empty answer, so a completed two-model comparison is
still pending. An execution error is not a model-quality score.

**Next: chat — compare Python and LangChain**

```powershell
uv run --extra advanced evals --config advanced/configs/openrouter-local.json suite chat --mode live
```

This sends a two-turn conversation through each implementation. Check whether
the second answer identifies the backpack and its INR 800 price. Review the
outputs yourself: the current Advanced chat workflow does not automatically
grade that answer. The configured free model may change availability.
For each new run, open `reports/advanced/RUN_ID/events.jsonl` for the messages
and response for every call. In Phoenix, expand the trace and select the
`chat.python.turn.1`, `chat.python.turn.2`, `chat.langchain.turn.1` or
`chat.langchain.turn.2` span to inspect its input and output. These logs contain
fictional shop content; do not use this exercise with real customer data.

### All available experiments

Use one experiment at a time. Replace `TOPIC` below with a name from the table.
Offline mechanisms make no model calls; they can still use real local protocols.

For live topics with OpenRouter, use the configured free model:

```powershell
uv run --extra advanced evals --config advanced/configs/openrouter-local.json suite TOPIC --mode live
```

Replace `TOPIC` with `judges`, `chat`, `rag`, `tools`, `agents`, `security`,
`operations`, `bias`, or `application`. The benchmark uses the two-model command
above. Some topics also require native tool calling or structured JSON; choose
an OpenRouter model that supports the needed feature. Free model availability
and responses can vary. Default RAG retrieval is local lexical retrieval;
the optional semantic embedding path uses a local Ollama embedding model.

For each live run, `reports/advanced/RUN_ID/events.jsonl` contains
`model.call.started`, `model.call.completed`, and, if needed,
`model.call.failed` events. Match their `span_id` to pair an input with its
output or error. The start event includes the messages, tool definitions and
JSON schema when supplied; the completion includes the response and tool calls.
Host tool executions also have `tool.call.started`, `tool.call.completed`, or
`tool.call.failed` events, including their arguments and results when available.
Phoenix displays the same captured input and output in each `model.generate`
span. These logs contain fictional inputs and generated outputs. Authentication
headers are not logged; use this feature only with synthetic learning data.
Offline topics do not make model calls, so they have no model-call logs.

In Phoenix, start at the `experiment.TOPIC` trace and expand its children:

| Span | What to inspect |
|---|---|
| `model.generate` | Actual request messages, model response or error, and usage. |
| `retrieval` | Question, retrieval variant, and ranked DemoShop documents. |
| `tool.*` / `mcp.call.*` | Tool arguments, outcome and access decision. |
| `agent.phase.*` | State at the start of each agent phase; model and tool calls have their own spans. |
| `evaluation.case.*` / `evaluation.benchmark.case` | Case output and checks, recorded after evaluation. |
| `evaluation.result` | Summary of the experiment, recorded after evaluation. |

The case, phase, and evaluation summary snapshots are recorded after or at the
start of a step; their span durations do not measure that step. Actual model,
retrieval, tool and MCP call spans cover their operations. The standalone
`annotations`, `compare`, and `regression --variant` commands read or calculate
locally and do not export their own traces; use `suite regression --mode offline`
to see traced regression examples. Human-judge generation and judging have
traces; the interactive human review and saved comparison are local records.

```powershell
uv run --extra advanced evals suite TOPIC --mode offline
uv run --extra advanced evals suite TOPIC --mode live
```

| Topics | Experiment | What it checks |
|---|---|---|
| 01 Foundations | `foundations` / offline | Exact/normalized scores and false passes/failures on authored outputs |
| 02 Deterministic | `regression` / offline | Catalog, policy, retrieval and permission expectations |
| 03 Human | `annotations` commands below | Real review records; labels start empty |
| 04 Judges | `judges`, `bias` / live | Python judge vs actual DeepEval; swapped answer order |
| 05 Benchmarking | `benchmark` / live | Inspect on two models, with Python price checks on the same answers |
| 06 Chat | `chat` / live | Python vs LangChain two-turn product memory |
| 07 RAG | `retrieval` / offline; `rag` / live | Python/Ragas recall, MRR/nDCG, retrieval and separate citation checks |
| 08 Tools | `tools` / live | Actual native calls, arguments, tool results and final answer |
| 09 MCP | `mcp` / offline | Real SDK stdio connection, schemas, state changes and denials |
| 10 Skills | `skills` / offline | Actual Markdown, hashes, selection precision/recall and false triggers |
| 11–12 Agents | `agents` / live | Python/LangGraph, single/multi state outcomes and handoff overhead |
| 13 Security | `security` / live; `promptfoo` / offline | Model workflow boundaries and actual domain permission probes |
| 14 Reliability | `statistics` / offline | Hand-checked Wilson interval and paired case-cluster bootstrap |
| 15 Operations | `operations` / live | Observed latency, calls, nullable tokens and unmeasured cost |
| 16 Regression | `regression` command below | Baseline succeeds; four controlled defects are detected locally |

To test how a model selects an MCP tool from descriptions, run:

```powershell
uv run --extra advanced evals --config advanced/configs/openrouter-local.json suite mcp-selection --mode live
```

The local MCP server advertises its actual `get_order` and `cancel_order`
schemas. The same status question is tested twice: once with the server's
descriptions and once with deliberately misleading descriptions supplied only
to the model. The server itself is not changed. Each case reports the expected
and selected tool and arguments, whether descriptions match the expected text,
and the MCP outcome when the correct read-only call is made. Wrong or mutating
choices are recorded but not executed. Inspect `mcp.selection.normal`,
`mcp.selection.misleading`, `model.generate`, and `mcp.call.get_order` in
Phoenix. This is one example, not a general tool-selection benchmark.

For an OpenRouter benchmark, select two model IDs available to your account:

```powershell
uv run --extra advanced evals suite benchmark --mode live --provider openrouter --models MODEL_A MODEL_B
```

The same three product-price questions go to both models. Inspect checks the
exact requested format; a Python check separately checks the price. This small
example demonstrates comparison, not general model quality. Explicit model IDs
make the comparison easier to interpret than an automatic router alias.

The judge needs a larger explicit budget than a subject example:

```powershell
uv run --extra advanced evals --config advanced/configs/judge.json suite judges --mode live
```

The two roles can use the same model; this is disclosed and does not establish
independence. Judge models are independently selectable through configuration.

## Run a suite, inspect evidence

There are two separate datasets: 30 authored matching controls in
[evaluation-v1.json](datasets/evaluation-v1.json), and 30 application scenarios in
[scenarios-v1.json](datasets/scenarios-v1.json). Each has a 20 development / 10
held-out split. Expectations stay out of application prompts.

```powershell
uv run --extra advanced evals suite application --mode live --split development
uv run --extra advanced evals report latest
uv run --extra advanced evals suite application --mode live --split held_out
uv run --extra advanced evals replay latest
uv run --extra advanced evals compare LEFT_RUN_ID RIGHT_RUN_ID
```

Compare matching tasks and splits. The comparison command rejects mismatched
tasks/splits and displays results for inspection; it does not establish causality.
Keep held-out cases for evaluation, rather than tuning prompts to their failures.

Reports distinguish execution status, narrow criterion verdict and review gaps.
Correct paraphrases can need review. RAG retrieval/citation success does not
establish answer faithfulness. An agent's state outcome does not establish that
its response and entire trajectory are correct. Malformed handoffs remain
execution errors. Inspect failing cases as well as passing summaries.

## Human review and local gates

For a connected OpenRouter exercise, generate live answers, review them blind,
then judge and compare the same saved answers. Add `OPENROUTER_API_KEY` and a
model available to your account as `OPENROUTER_MODEL` in the root `.env` first.
Choose a free model ID if desired; the judge endpoint must support JSON schema.

```powershell
uv run --extra advanced evals human-judge generate
# Copy the run ID printed above into the next three commands.
uv run --extra advanced evals human-judge review RUN_ID
uv run --extra advanced evals human-judge judge RUN_ID
uv run --extra advanced evals human-judge compare RUN_ID
```

The run adds one clearly labeled authored wrong-price answer as a control, in
addition to the requested model answers. You review it too; the judge does not
receive the control's expected label. The comparison reports whether the judge
detected the wrong answer separately from agreement on model-generated answers.
Review asks for your label and reason before the judge stage is available.
Pass `--model MODEL_ID` to `generate` or `judge` to select a different model;
`OPENROUTER_JUDGE_MODEL` is also supported for the judge. Generation and judge
failures remain errors, and `uncertain` labels are excluded from the agreement
denominator. This five-case, one-reviewer exercise does not establish judge
reliability. This synthetic exercise also shows its question and generated answer
in Phoenix, and the judge result after review. Do not use real customer data.
Reports and traces stay in ignored `reports/advanced/`. To show an older saved
answer in Phoenix without another model call, run `evals human-judge trace RUN_ID`;
the resulting trace is marked as a replay.

```powershell
uv run --extra advanced evals annotations export reports/advanced/review.json
# People independently fill the annotations array using the exported rubric.
uv run --extra advanced evals annotations import reports/advanced/review.json
uv run --extra advanced evals regression --variant baseline
uv run --extra advanced evals regression --variant wrong-price
```

The last command deliberately exits 1. Other defects: `wrong-policy`,
`wrong-retrieval`, `wrong-authorization`. These are labeled mutation controls;
unsafe permissions are never installed in the domain tools. Reviewer agreement
and kappa are available in `evaluation.agreement`; independent labels and
adjudication must come from people. No learner mastery is inferred.

For the actual local promptfoo trial, install its pinned local package once:

```powershell
npm ci --prefix advanced --no-audit --no-fund
uv run --extra advanced evals suite promptfoo --mode offline
```

This invokes real Python domain tools on authored requests, without hosted
red-team generators. Its pass rate measures those boundary checks, not general
model jailbreak resistance. Inference remains Ollama/local or explicit OpenRouter;
LangSmith and unsolicited framework telemetry are disabled.

## Implementation and limits

Start with [LEARNING.md](LEARNING.md), then inspect
[the Python package](src/ai_quality_evals_advanced/) and [tests](tests/).
Actual procedures are [cancellation](skills/cancel-owned-order/SKILL.md) and
[returns](skills/return-owned-order/SKILL.md). The real MCP server is
[mcp_server.py](src/ai_quality_evals_advanced/mcp_server.py).

State uses isolated SQLite sandboxes. Schemas, ownership, policies, idempotency,
call budgets and deadlines are host-enforced. OpenTelemetry emits correlated
model/tool/role/grader traces to native Phoenix. The human–judge learning exercise
also captures synthetic questions and generated answers as described above.
Atomic reports include configuration and fixture/lock hashes; no old evidence is
rewritten. Generated evidence stays ignored under root `reports/advanced/`.

Semantic retrieval requires `retrieval: "semantic"` and an explicitly configured,
already installed `embedding_model`. Missing models stop the run; no download or
lexical fallback. The current machine has no dedicated embedding model, so that
live verification remains pending. Some OpenRouter workflows have been exercised
live; verification across all workflows, broader human calibration, security
coverage and learner review remain open.
Small experiments and passing tests do not establish production reliability.

```powershell
uv run --locked --extra dev --extra advanced pytest
uv run --locked --extra dev --extra advanced ruff check .
uv run --locked --extra dev --extra advanced ruff format --check .
```

Offline tests use no credentials or model service. Phoenix/DeepEval automatic
pytest plugins are disabled. Third-party Python 3.14 deprecation warnings remain
visible. No Docker, custom UI, CI, deployment, model downloads or hosted tracing.
