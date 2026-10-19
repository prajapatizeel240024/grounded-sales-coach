"""Settings, read from the environment (and an optional .env file at the repo root)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def _find_root() -> Path:
    env = os.environ.get("FIELDCOACH_ROOT")
    if env:
        return Path(env).resolve()
    candidate = PACKAGE_DIR.parent  # editable install / running from a checkout
    if (candidate / "data" / "transcripts").is_dir():
        return candidate
    return Path.cwd()


def load_dotenv(path: Path) -> None:
    """Tiny .env reader: KEY=VALUE lines. Never overrides variables that are already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    root: Path
    model: str
    effort: str | None
    max_tokens: int
    concurrency: int
    database_url: str | None
    anthropic_api_key: str | None

    @property
    def transcripts_dir(self) -> Path:
        return self.root / "data" / "transcripts"

    @property
    def runs_dir(self) -> Path:
        return self.root / "data" / "runs"

    @property
    def golden_path(self) -> Path:
        return self.root / "evals" / "golden.json"

    @property
    def results_dir(self) -> Path:
        return self.root / "evals" / "results"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache" / "llm"

    @property
    def readme_path(self) -> Path:
        return self.root / "README.md"


def get_settings() -> Settings:
    root = _find_root()
    load_dotenv(root / ".env")
    return Settings(
        root=root,
        model=os.environ.get("FIELDCOACH_MODEL") or "claude-sonnet-5-5",
        effort=os.environ.get("FIELDCOACH_EFFORT") or None,
        max_tokens=int(os.environ.get("FIELDCOACH_MAX_TOKENS") or 16000),
        concurrency=max(1, int(os.environ.get("FIELDCOACH_CONCURRENCY") or 4)),
        database_url=os.environ.get("DATABASE_URL") or None,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
    )
