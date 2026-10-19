"""Thin Claude client: structured outputs validated by Pydantic, with a disk cache.

Requests use `output_config.format` with a JSON schema built from the Pydantic model by
`anthropic.transform_schema`. No sampling parameters are sent, because current models only
accept their defaults. Responses are cached under a hash of model, effort, prompt, input,
schema and sample tag, so rerunning an unchanged eval is free and any prompt edit naturally
misses the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    import anthropic

    return anthropic.transform_schema(model)


class ClaudeClient:
    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        effort: str | None = None,
        cache_dir: Path | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, max_retries=4, timeout=600.0)
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.cache_dir = cache_dir
        self.usage = Usage()
        self._lock = threading.Lock()

    def structured(self, *, system: str, user: str, output: type[M], tag: str) -> M:
        """One request whose reply must parse into `output`. Retries once on invalid JSON."""
        schema = output_schema(output)
        key = hashlib.sha256(
            json.dumps(
                {"model": self.model, "effort": self.effort, "system": system,
                 "user": user, "schema": schema, "tag": tag},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:32]

        cached = self._cache_get(key)
        if cached is not None:
            try:
                result = output.model_validate(cached)
            except ValidationError:
                pass  # stale entry from an older schema; make a fresh call
            else:
                with self._lock:
                    self.usage.cache_hits += 1
                return result

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort

        problem: str | None = None
        for attempt in range(2):
            content = user
            if problem:
                content += (
                    f"\n\n<previous_attempt_error>\n{problem}\n</previous_attempt_error>\n"
                    "Your previous reply did not match the schema. Reply again with valid JSON."
                )
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config=output_config,
            )
            self._record(response, retry=attempt > 0)
            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                raise LLMError(f"{tag}: the model declined to answer")
            if stop == "max_tokens":
                raise LLMError(
                    f"{tag}: reply was cut off at max_tokens={self.max_tokens}; "
                    "raise FIELDCOACH_MAX_TOKENS"
                )
            text = "".join(
                getattr(block, "text", "") for block in response.content
                if getattr(block, "type", None) == "text"
            ).strip()
            try:
                data = json.loads(text)
                result = output.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                problem = f"{type(exc).__name__}: {str(exc)[:800]}"
                continue
            self._cache_put(key, tag, data)
            return result
        raise LLMError(f"{tag}: no valid JSON after 2 attempts ({problem})")

    # ------------------------------------------------------------------ internals

    def _record(self, response: Any, retry: bool) -> None:
        usage = getattr(response, "usage", None)
        with self._lock:
            self.usage.calls += 1
            self.usage.retries += int(retry)
            if usage is not None:
                self.usage.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.usage.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key: str) -> Any:
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["output"]
        except (OSError, ValueError, KeyError):
            return None

    def _cache_put(self, key: str, tag: str, data: Any) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"tag": tag, "model": self.model, "output": data}), encoding="utf-8")
        os.replace(tmp, path)
