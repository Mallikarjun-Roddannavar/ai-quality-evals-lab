"""Native Phoenix startup with workspace-local storage and no Docker."""

import os
import subprocess
import sys
from urllib.parse import urlsplit

from ai_quality_evals_advanced.contracts import ARTIFACTS, ROOT, Settings


def phoenix_environment(config: Settings) -> dict[str, str]:
    folder = ARTIFACTS / "phoenix"
    folder.mkdir(parents=True, exist_ok=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith(("API_KEY", "ACCESS_KEY", "SECRET_KEY"))
        and not key.startswith(("LANGSMITH_", "LANGCHAIN_", "OTEL_", "PHOENIX_"))
    }
    env.update(
        {
            "PHOENIX_WORKING_DIR": str(folder),
            "PHOENIX_SQL_DATABASE_URL": "sqlite:///"
            + (folder / "phoenix.db").as_posix(),
            "PHOENIX_ALLOW_EXTERNAL_RESOURCES": "false",
            "PHOENIX_HOST": "127.0.0.1",
            "PHOENIX_PORT": str(urlsplit(config.phoenix_endpoint).port or 6006),
            "PHOENIX_DEFAULT_RETENTION_POLICY_DAYS": "30",
            "LANGSMITH_TRACING": "false",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    return env


def serve(config: Settings) -> int:
    child = subprocess.Popen(
        [sys.executable, "-m", "phoenix.server.main", "serve"],
        cwd=ROOT,
        env=phoenix_environment(config),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        return child.wait()
    except KeyboardInterrupt:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        return 130
