# Intro repository instructions

## Scope

- This repository snapshot contains the runnable DemoShop Intro lab. The parent
  planning workspace remains private and is not included here.
- Keep Advanced experiments and deferred learning notes/blog assets out of this
  snapshot; their local copies may remain in the workspace.
- Use the same synthetic DemoShop facts across topics. Do not add real customer,
  order, payment, or enterprise data.
- Keep starter-demo results distinct from full module validation and learner
  understanding.

## Development

- Keep the root `pyproject.toml` and `uv.lock` as the single environment for Intro.
- Use only OpenRouter or local Ollama models for live inference. Never add direct
  OpenAI or OCI inference calls.
- Offline tests must not require credentials, network access, or a live model.
- Run commands from this repository root. Check tests, Ruff lint, and formatting
  before describing a code change as validated.
- Keep `.env`, credentials, generated reports, caches, virtual environments, and
  temporary artifacts out of Git. Never weaken ignore rules to include them.
- Do not initialize or alter Git metadata, stage, commit, or push unless the user
  explicitly authorizes that specific action.
