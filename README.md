# AI Evaluation Lab

## AI quality in the era of LLM apps and agents

Today's AI products combine language models with prompts, retrieval, tools,
conversation history, and sometimes multiple agents. A fluent answer can still
be wrong, unsupported, unsafe, or expensive. Evaluation helps teams check the
behavior that matters before and after they change a model or application.

DemoShop is a small, fictional shop used to explore those checks in one place.
Its browser lab offers sixteen starter demonstrations, including answer
correctness, human and model review, model comparison, chat, retrieval, tool
use, MCP, application skills, agents, security, statistics, operations, and
regression checks. The examples connect current AI engineering patterns to
concrete questions: Did the model use the facts? Did the right tool run? Is the
result supported? What changed across models or runs?

## Run the lab

From this repository root, with Python and `uv` installed:

```powershell
uv sync --locked --extra dev
uv run lab
```

Open http://127.0.0.1:8501 and choose a topic. Live model examples use Ollama
or OpenRouter. Offline tests do not need a model service or API key.

See [the Intro guide](intro/README.md) for provider setup, replay, and result
interpretation. These are starter demonstrations, not a certification of a
model or a complete production evaluation suite. PASS, FAIL, and NEEDS REVIEW
apply only to the specific check shown in each demo; model-graded results still
need human scrutiny.

## Offline checks

```powershell
uv run --locked --offline pytest
uv run --locked --offline ruff check .
uv run --locked --offline ruff format --check .
uv run lab --check
uv run lab --check --broken
```

The final command intentionally exits nonzero for a seeded price regression.
Generated evidence stays in ignored root `reports/`.
