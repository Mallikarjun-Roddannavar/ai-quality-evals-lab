# DemoShop repository instructions

## Scope

- This repository contains the runnable Intro lab and Advanced experiments.
  The parent planning workspace remains private and is not included here.
- Keep deferred learning notes, blogs and Intro blog assets out of the snapshot;
  their local copies may remain in the workspace.
- Use the same synthetic DemoShop facts across topics. Do not add real customer,
  order, payment, or enterprise data.
- Keep starter-demo results distinct from full module validation and learner
  understanding.

## Development

- Keep the root `pyproject.toml` and `uv.lock` as the shared environment for
  Intro and the optional Advanced dependencies.
- Use only OpenRouter or local Ollama models for live inference. Never add direct
  OpenAI or OCI inference calls.
- Offline tests must not require credentials, network access, or a live model.
- Run commands from this repository root. Check tests, Ruff lint, and formatting
  before describing a code change as validated.
- Keep credentials, generated reports, caches, virtual environments, and
  temporary artifacts out of Git. The shareable `.env` must contain no secrets.
- Do not initialize or alter Git metadata, stage, commit, or push unless the user
  explicitly authorizes that specific action.
