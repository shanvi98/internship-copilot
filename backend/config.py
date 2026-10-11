"""
Runtime configuration, read once from environment variables.

Everything here has a sensible default so the app runs with zero setup; the
env vars exist so a deployment (Docker, Render, a teammate's laptop) can point
at a different Ollama host/model or Claude model without code changes.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"))
    ollama_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    anthropic_model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5"))
    # low | medium | high | xhigh | max — set explicitly because Opus 5.5's default is medium
    anthropic_effort: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_EFFORT", "medium"))
    llm_timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 60.0))
    db_path: Path = field(
        default_factory=lambda: Path(os.environ.get("COPILOT_DB_PATH", Path(__file__).parent / "copilot.db"))
    )
    cors_origins: list[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS", "*"))
    # Fetching arbitrary URLs from a public server is an SSRF surface; it can be switched off entirely.
    allow_url_fetch: bool = field(default_factory=lambda: os.environ.get("ALLOW_URL_FETCH", "1") not in ("0", "false", "no"))


settings = Settings()
