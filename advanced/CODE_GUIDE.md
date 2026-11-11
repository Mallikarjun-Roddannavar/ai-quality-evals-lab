# Code and libraries at a glance

Use this guide to find your way around the Intro and Advanced code. You do not
need to learn every library before running an example. For Foundations, start
with `argparse`, `json`, Pydantic, and the matching functions.

## Follow an Advanced run

```text
CLI command → cli.py → experiments.py → selected experiment
            → application or prepared cases → evaluation → result and trace
```

For `evals suite foundations --mode offline`, the CLI selects Foundations,
`evaluation.py` loads prepared cases, and the matching functions compare each
answer with its reference. There is no model call in this experiment.

| Code | Responsibility |
|---|---|
| [cli.py](src/ai_quality_evals_advanced/cli.py) | Reads commands and arguments; selects what to run. |
| [experiments.py](src/ai_quality_evals_advanced/experiments.py) | Routes to an experiment and gathers its result. |
| [evaluation.py](src/ai_quality_evals_advanced/evaluation.py) | Loads evaluation cases and calculates matching and human-review results. |
| [matching.py](../intro/src/ai_quality_evals/evaluators/matching.py) | Implements exact and normalized text comparisons. |
| [providers.py](src/ai_quality_evals_advanced/providers.py) | Sends live model requests to OpenRouter or Ollama. |
| [domain.py](src/ai_quality_evals_advanced/domain.py) | Implements the fictional shop and sandbox order operations. |
| [applications.py](src/ai_quality_evals_advanced/applications.py) | Builds chat, retrieval, tool and skill application behavior. |
| [agents.py](src/ai_quality_evals_advanced/agents.py) | Implements agent workflows and handoffs. |
| [frameworks.py](src/ai_quality_evals_advanced/frameworks.py) | Connects experiments to frameworks including Ragas and Inspect. |
| [telemetry.py](src/ai_quality_evals_advanced/telemetry.py) | Creates and exports execution traces. |
| [lab.py](../intro/src/ai_quality_evals/lab.py) | Builds the Intro Streamlit interface. |

## External packages and tools

These are direct project dependencies in [pyproject.toml](../pyproject.toml),
plus promptfoo in [package.json](package.json). Most Advanced packages are
installed through the optional `advanced` extra; they are not all used in each run.

| Package or tool | Purpose here |
|---|---|
| Pydantic | Defines and validates data such as cases, configuration and model responses. |
| Streamlit | Builds the Intro browser interface. |
| MCP SDK | Runs local Model Context Protocol clients and servers. |
| DeepEval | Demonstrates LLM-based faithfulness judging. |
| Ragas | Demonstrates retrieval metrics, including document-ID recall. |
| Inspect AI | Runs benchmark tasks with models and scorers. |
| LangChain | Builds chat and RAG application variants to evaluate. |
| LangGraph | Builds stateful single-agent and multi-agent workflows. |
| OpenTelemetry SDK | Creates traces and spans describing execution steps. |
| OpenTelemetry HTTP exporter | Sends traces to a configured endpoint. |
| Arize Phoenix | Stores and displays local traces. |
| promptfoo | Runs authored local security assertions using Node.js. |
| pytest | Runs automated Python tests. |
| Ruff | Checks and formats Python code. |
| setuptools | Builds and installs the Python package and its CLI commands. |

`langchain-ollama` and `langchain-community` are also declared, but the current
project source has no direct imports of them. `requests` is imported by
telemetry and currently arrives as a transitive dependency. `uv.lock` and
`package-lock.json` also contain dependencies of these packages; installation
alone does not mean the project directly uses each one.

LangChain and LangGraph build workflows; they do not grade correctness. Phoenix
shows execution; a trace alone does not establish correctness. Framework scores
measure only the criteria configured for the experiment.

## Python built-in modules

These are included with Python; they do not need separate installation.

| Module | Use in this project |
|---|---|
| `argparse` | Read CLI commands and options. |
| `json` | Read datasets and write structured results. |
| `pathlib` | Work with file and directory paths. |
| `typing` | Describe expected types and allowed values. |
| `os`, `sys` | Access environment settings, process arguments and runtime information. |
| `datetime`, `time` | Record timestamps and measure durations. |
| `uuid`, `hashlib` | Create run IDs and hashes for provenance. |
| `re` | Match and validate text with regular expressions. |
| `math`, `statistics`, `random`, `decimal` | Perform numeric and statistical calculations. |
| `collections` | Count items and organize data. |
| `sqlite3` | Store Advanced sandbox shop data locally. |
| `http`, `urllib` | Make HTTP requests to models and local services. |
| `asyncio`, `threading`, `concurrent.futures` | Handle asynchronous and concurrent work. |
| `contextlib`, `contextvars` | Manage context, including trace context. |
| `subprocess`, `shutil` | Start or locate external processes and tools. |
| `importlib`, `platform` | Inspect package versions and runtime environment. |
| `copy`, `ast`, `inspect` | Copy data and inspect Python objects or code. |
| `gzip`, `io`, `socket`, `types` | Support specific test fixtures and utilities. |

## Reading an import

```python
from ai_quality_evals_advanced.evaluation import controls
```

This imports `controls` from **our own** `evaluation.py` file. In contrast,
`from pydantic import Field` imports from an external installed package, and
`import json` imports a module included with Python.
