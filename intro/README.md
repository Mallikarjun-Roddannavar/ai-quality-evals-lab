# Intro: one DemoShop, 16 starter examples

The browser lab covers chat, RAG, tools, MCP, application skills and bounded
single/multi-agent examples. Choose a topic and try it in the browser.

## Start from the repository root

```powershell
uv sync --locked --extra dev
uv run lab
```

Open http://127.0.0.1:8501, choose Foundations and click Run demo. Read the facts,
inspect the actual answer and explain what its score means. Then try Deterministic
or enter your own assessment under Human evaluation. Calls occur only on Run;
Replay latest saved run makes no new model call.

Model-based examples support either local Ollama or OpenRouter. Ollama needs an
already installed model; defaults are gemma4:e2b, with gemma2:2b for comparison.
OpenRouter needs an API key in `OPENROUTER_API_KEY` and a model ID available to
your account. The repository-root `.env` file is loaded automatically; optional
`OPENROUTER_MODEL`, `OPENROUTER_SECONDARY_MODEL` and `OPENROUTER_JUDGE_MODEL`
values set its initial model selections.
Choose the provider and model IDs under Advanced settings. The key is read only
from `.env` or the process environment and is never written to run reports. Hosted model
calls may incur charges; only click Run demo when you intend to make a live call.
When OpenRouter returns a usage cost, the lab records and displays the
provider-reported USD amount; missing cost remains unknown. Local energy cost is
not measured.
Topics that use structured JSON or native tool calling require a model endpoint
that supports those request parameters; the lab asks OpenRouter to route only to
compatible endpoints for those requests.
No models are downloaded automatically. MCP, security, statistics and regression
examples do not need an LLM. Skills execution uses the selected live provider.

For OpenRouter, edit `.env` and add your key to `OPENROUTER_API_KEY`. If `.env` is
missing, copy the template first. Save the file, then launch or restart Streamlit
from the repository root:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
# After saving and closing the editor:
uv run lab
```

The `.env` file is ignored by Git. PowerShell environment variables override its
values. Never paste the key into the browser, source files, or saved reports.

## Results and evidence

**Pass/fail** refers only to the stated check. **Needs review** does not mean an
answer is automatically wrong. **Execution error** means the demonstration did
not complete; it is never replaced silently by a replay or fake answer.

The displayed evidence label distinguishes actual model output, controlled local
examples and replay. A replay keeps its original timestamp/model metadata. Human
assessments are saved separately, never inserted as manufactured ground truth.

The single-agent demo has at most four model calls and three tool calls. The
comparison uses two independent runs, each limited to two model calls/three tool
calls (at most four model calls in total). Every run resets the in-memory orders.
Ownership and shipped-order restrictions are enforced in tools. There are no real
payments, customer records or order integrations.

The first live sweep showed a fallible judge: it labeled a source-appropriate
"Unknown" shipping answer incorrect. Topic 04 deliberately leaves the judgment
for review; agreement/calibration are not assumed. Topic 07 checks retrieval and
shows citations but does not automatically prove every generated claim is grounded.
Topic 10 uses a small host router and bundled skill loader, not a general third-party
skill platform. Topic 14's constructed examples violate
population-sampling assumptions: its interval teaches the formula, not model accuracy.

## Current examples and limits

| Topic | Actual mechanism | Small-scope limitation |
|---|---|---|
| 01 Foundations | Existing Python exact/normalized evaluators score actual model text. | Wording equality is not factual correctness. |
| 02 Deterministic | Existing bounded price parser compares an answer with the catalog. | Unsupported wording needs review. |
| 03 Human | Browser form accepts a person's name, label and explanation; writes a separate annotation. | No independent agreement study or fabricated labels. |
| 04 Judges | A real local model receives source facts and a rubric and returns a validated judgment. | Uncalibrated; same-model use is disclosed. |
| 05 Benchmarking | Two installed models answer the same three questions with recorded usage. | Development comparison, not a statistically established winner. |
| 06 Chat | Two real calls carry user/assistant history to the second turn. | One memory case, not a conversation benchmark. |
| 07 RAG | Keyword retrieval ranks three short documents derived from the shop fixture; only retrieved context is supplied to generation. | Real lexical retrieval without embeddings; citation presence is not claim grounding. |
| 08 Tool calling | Native Ollama tools/tool_calls messages, validated dispatch and tool-role feedback. | One public lookup; no JSON-action fallback for unsupported models. |
| 09 MCP | Dedicated SDK server process and SDK stdio client perform discovery and tool calls. | Local protocol, not production identity/OAuth. |
| 10 Skills | Read real SKILL.md metadata/body, select for a matching request, inject instructions and execute sandbox cancellation. | Host routing is intentionally small; no arbitrary scripts or external skill marketplace. |
| 11 Single agent | Model selects bounded JSON actions, receives real tool results and changes an owned sandbox order. | Constrained action loop, not an unrestricted autonomous planner. |
| 12 Multi-agent | Actual policy-reviewer model call, recorded handoff and order-worker model call; compare with fresh single-agent state. | Same model with distinct roles; no claim that extra coordination improves quality. |
| 13 Security | Real tool-side ownership checks reject cross-customer reads/writes while legitimate access works. | Controlled synthetic security test, not comprehensive red teaming. |
| 14 Statistics | Actual count/rate/Wilson calculations on explicitly authored answers. | Formula demonstration, not a valid sampled model-performance estimate. |
| 15 Operations | Read actual provider latency and token counters from run responses. | Cost unmeasured; one call cannot establish p95. |
| 16 Regression | Execute catalog checks on correct and deliberately wrong-price variants; CLI exits nonzero for the defect. | Working local gate; no remote CI service is configured. |

## Code map

- [lab.py](src/ai_quality_evals/lab.py): shared page and human annotation form.
- [lab_evals.py](src/ai_quality_evals/lab_evals.py): each example, scoring, result schema, provenance and replay.
- [lab_systems.py](src/ai_quality_evals/lab_systems.py): retrieval, skill loading, tools, agent loops and MCP client.
- [mcp_server.py](src/ai_quality_evals/mcp_server.py): standalone MCP server and registered lookup tool.
- [SKILL.md](skills/cancel-owned-order/SKILL.md): cancellation instructions actually read by the app.
- [ollama.py](src/ai_quality_evals/providers/ollama.py): native text/tool HTTP requests and actual raw responses.
- [test_lab.py](tests/test_lab.py): offline model doubles, real MCP connection, state/permissions, skill loading and browser rerender checks.

Tests use clearly authored doubles and make no live model calls. Live runs use the
selected Ollama or OpenRouter provider. Application skills live in `skills/` and
are separate from coding-assistant developer skills.

## Developer checks

Run all commands from the repository root:

```powershell
uv run --locked --offline pytest
uv run --locked --offline ruff check .
uv run --locked --offline ruff format --check .
uv run lab --check
uv run lab --check --broken
```

The last command intentionally exits 1 for a controlled regression; the normal
check exits 0. Offline tests use model doubles and a genuine local MCP connection.
Generated evidence stays in ignored root `reports/`.
The standalone MCP server is `uv run python -m ai_quality_evals.mcp_server`.
Streamlit 1.66.1 and MCP 2.3.0 remain pinned; no new framework is added.
