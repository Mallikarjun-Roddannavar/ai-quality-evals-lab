# Intro: one DemoShop, 16 starter examples

The browser lab covers chat, RAG, tools, MCP, application skills and bounded
single/multi-agent examples. Choose a topic and try it in the browser.

## Start from the repository root

```powershell
uv sync --locked --extra dev
uv run lab
```

Open http://127.0.0.1:8501 and choose a topic. To start without a model, select
Security or Regression / CI and click Run demo. To try Foundations, first set
up a model provider below; then read the facts, inspect the answer and explain
what its score means. Model calls occur only on Run; Replay latest saved run
makes no new model call and requires a previously saved run.

### Which topics need a model?

| Requirement | Topics |
|---|---|
| No model or API key | 09 MCP, 13 Security, 14 Statistical reliability, 16 Regression / CI |
| OpenRouter or Ollama | 01 Foundations, 02 Deterministic, 03 Human evaluation, 04 LLM judges, 05 Model benchmarking, 06 Chat, 07 RAG, 08 Tool calling, 10 Skills, 11 Single agent, 12 Multi-agent, 15 Operations |

The deterministic and human evaluation demos use a model to generate the answer
before it is checked or reviewed. The MCP demo starts a local server process.

To run model examples, add your OpenRouter API key to the project-root `.env`
file. In the lab, open **Advanced settings**, select OpenRouter, and choose an
available model ending in `:free`. Model benchmarking uses two selected models;
the judge demo also lets you select a judge model. Some demos need a model that
supports structured output or tool calling.

Ollama is also supported if you prefer a locally installed model.

Open `.env`, add `OPENROUTER_API_KEY=your-key`, and save it. Create the file if
it is missing. Then start or restart the lab from the repository root:

```powershell
notepad .env
# After saving and closing the editor:
uv run lab
```

The `.env` file can be tracked by Git. Remove your key before staging or
committing it. PowerShell environment variables override its values. Never paste
the key into the browser, source files, or saved reports.

## Reading results

- **Pass / fail:** The result passed or failed the stated check.
- **Needs review:** Inspect the result yourself; it is not necessarily wrong.
- **Execution error:** The demo did not complete.

The lab labels live model output, controlled examples, and replays. A model
judge can be wrong, and a citation alone does not prove an answer is supported.
The shop and its orders are fictional; agent runs start with fresh sandbox state.

## Current examples and limits

| Topic | Actual mechanism | Small-scope limitation |
|---|---|---|
| 01 Foundations | Existing Python exact/normalized evaluators score actual model text. | Wording equality is not factual correctness. |
| 02 Deterministic | Existing bounded price parser compares an answer with the catalog. | Unsupported wording needs review. |
| 03 Human | Browser form accepts a person's name, label and explanation; writes a separate annotation. | No independent agreement study or fabricated labels. |
| 04 Judges | The selected model receives source facts and a rubric and returns a validated judgment. | Uncalibrated; same-model use is disclosed. |
| 05 Benchmarking | Two selected models answer the same three questions with recorded usage. | Development comparison, not a statistically established winner. |
| 06 Chat | Two real calls carry user/assistant history to the second turn. | One memory case, not a conversation benchmark. |
| 07 RAG | Keyword retrieval ranks three short documents derived from the shop fixture; only retrieved context is supplied to generation. | Real lexical retrieval without embeddings; citation presence is not claim grounding. |
| 08 Tool calling | Native tool calls, validated dispatch and tool-role feedback through the selected provider. | One public lookup; no JSON-action fallback for unsupported models. |
| 09 MCP | Dedicated SDK server process and SDK stdio client perform discovery and tool calls. | Local protocol, not production identity/OAuth. |
| 10 Skills | Read real SKILL.md metadata/body, select for a matching request, inject instructions and execute sandbox cancellation. | Host routing is intentionally small; no arbitrary scripts or external skill marketplace. |
| 11 Single agent | Model selects bounded JSON actions, receives real tool results and changes an owned sandbox order. | Constrained action loop, not an unrestricted autonomous planner. |
| 12 Multi-agent | Actual policy-reviewer model call, recorded handoff and order-worker model call; compare with fresh single-agent state. | The same model selection is used for distinct roles; a router may choose different underlying models. Extra coordination is not presumed better. |
| 13 Security | Real tool-side ownership checks reject cross-customer reads/writes while legitimate access works. | Controlled synthetic security test, not comprehensive red teaming. |
| 14 Statistics | Actual count/rate/Wilson calculations on explicitly authored answers. | Formula demonstration, not a valid sampled model-performance estimate. |
| 15 Operations | Read actual provider latency and token counters from run responses. | One call cannot establish p95. |
| 16 Regression | Execute catalog checks on correct and deliberately wrong-price variants; CLI exits nonzero for the defect. | Working local gate; no remote CI service is configured. |

## Code map

- [lab.py](src/ai_quality_evals/lab.py): shared page and human annotation form.
- [lab_evals.py](src/ai_quality_evals/lab_evals.py): each example, scoring, result schema, provenance and replay.
- [lab_systems.py](src/ai_quality_evals/lab_systems.py): retrieval, skill loading, tools, agent loops and MCP client.
- [mcp_server.py](src/ai_quality_evals/mcp_server.py): standalone MCP server and registered lookup tool.
- [SKILL.md](skills/cancel-owned-order/SKILL.md): cancellation instructions actually read by the app.
- [ollama.py](src/ai_quality_evals/providers/ollama.py): native text/tool HTTP requests and actual raw responses.
- [test_lab.py](tests/test_lab.py): offline model doubles, real MCP connection, state/permissions, skill loading and browser rerender checks.

Application skills live in `skills/` and are separate from coding-assistant
developer skills.

## Developer checks

Run all commands from the repository root:

```powershell
uv run --locked --offline pytest
uv run --locked --offline ruff check .
uv run --locked --offline ruff format --check .
uv run lab --check
uv run lab --check --broken
```

The last command intentionally fails to demonstrate a regression check.
