"""Shared fixtures. Nothing here needs an API key, a network or a database."""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import anthropic._base_client
import pytest

from fieldcoach.config import Settings
from fieldcoach.evals import load_golden
from fieldcoach.transcripts import load_transcripts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings() -> Settings:
    # A fake model name keeps any cache key used in tests apart from real runs.
    return Settings(
        root=ROOT,
        model="claude-test-model",
        effort=None,
        max_tokens=4000,
        concurrency=2,
        database_url=None,
        anthropic_api_key="test-key",
    )


@pytest.fixture(scope="session")
def transcripts(settings):
    return {t.id: t for t in load_transcripts(settings.transcripts_dir)}


@pytest.fixture(scope="session")
def golden(settings):
    return load_golden(settings.golden_path)


def message_json(payload, *, stop_reason: str = "end_turn", thinking: bool = True) -> dict:
    """A Messages API response body, shaped like the real thing (thinking block first)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": "Reading the turns.", "signature": "c2lnbmF0dXJl"})
    content.append({"type": "text", "text": text})
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test-model",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }


class FakeAnthropic:
    """The real anthropic.Anthropic client, wired to an in-process transport.

    `responder(body)` gets each request body as a dict and returns a response body dict, or a
    (status, body) tuple for errors. Every request body is kept in `.requests` so tests can check
    what the SDK sent.
    """

    def __init__(self, responder):
        self.requests: list[dict] = []
        # Recent SDKs ship on `httpx2`, older ones on `httpx`; use whichever this SDK imports.
        http = getattr(anthropic._base_client, "httpx2", None) or getattr(anthropic._base_client, "httpx")

        def handler(request):
            body = json.loads(request.content)
            self.requests.append(body)
            result = responder(body)
            status, payload = result if isinstance(result, tuple) else (200, result)
            return http.Response(status, json=payload)

        self.client = anthropic.Anthropic(
            api_key="test-key",
            base_url="https://api.anthropic.test",
            http_client=http.Client(transport=http.MockTransport(handler)),
            max_retries=0,
        )
