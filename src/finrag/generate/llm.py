"""LLM client: a small provider-agnostic interface, a Gemini implementation
(REST via httpx, no SDK), and a disk cache.

The cache matters on Gemini's free tier, where daily request caps can be low.
Re-running an eval with an unchanged prompt costs zero requests. Keys cover
everything that affects the output: model, system prompt, prompt, schema, and
temperature.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0  # includes "thinking" tokens on thinking models
    latency_s: float = 0.0
    cached: bool = False

    def json(self) -> dict:
        """Parse a JSON reply, tolerating a ```json ... ``` fence around it."""
        text = self.text.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        return json.loads(fenced.group(1) if fenced else text)


class LLMClient(Protocol):
    model: str

    def generate(
        self, system: str, prompt: str, schema: dict | None = None, temperature: float = 0.0
    ) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 5,
        max_output_tokens: int = 4096,  # thinking tokens count against this too
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set (get one at aistudio.google.com)")
        self.model, self.max_attempts, self.max_output_tokens = (
            model,
            max_attempts,
            max_output_tokens,
        )
        # Key goes in a header, not the URL, so it never lands in logs.
        self._http = httpx.Client(
            headers={"x-goog-api-key": api_key}, timeout=120.0, transport=transport
        )

    def generate(
        self, system: str, prompt: str, schema: dict | None = None, temperature: float = 0.0
    ) -> LLMResponse:
        config: dict = {"temperature": temperature, "maxOutputTokens": self.max_output_tokens}
        if schema:
            config |= {"responseMimeType": "application/json", "responseSchema": schema}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        started = time.time()
        data = self._post(body)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise LLMError(f"Gemini returned no answer ({reason})")
        parts = candidates[0].get("content", {}).get("parts", [])
        # Thinking models may return thought summaries as parts flagged thought=true.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            latency_s=time.time() - started,
        )

    def _post(self, body: dict) -> dict:
        """429 (free-tier rate limit) and 5xx are retried with backoff (2-30s)."""
        url = GEMINI_URL.format(model=self.model)
        for attempt in range(self.max_attempts):
            response = self._http.post(url, json=body)
            temporary = response.status_code == 429 or response.status_code >= 500
            if temporary and attempt < self.max_attempts - 1:
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            if response.is_error:
                raise LLMError(f"Gemini API {response.status_code}: {response.text[:600]}")
            return response.json()
        raise AssertionError("unreachable")


class CachedLLM:
    """Wraps any LLMClient with a SQLite response cache."""

    def __init__(self, inner: LLMClient, path: Path):
        self.inner, self.model = inner, inner.model
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, response TEXT)")

    @staticmethod
    def key(model: str, system: str, prompt: str, schema: dict | None, temperature: float) -> str:
        raw = json.dumps([model, system, prompt, schema, temperature], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def generate(
        self, system: str, prompt: str, schema: dict | None = None, temperature: float = 0.0
    ) -> LLMResponse:
        key = self.key(self.model, system, prompt, schema, temperature)
        row = self._db.execute("SELECT response FROM llm WHERE key = ?", (key,)).fetchone()
        if row:
            return LLMResponse(**json.loads(row[0]) | {"cached": True, "latency_s": 0.0})
        response = self.inner.generate(system, prompt, schema, temperature)
        self._db.execute(
            "INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, json.dumps(asdict(response)))
        )
        self._db.commit()
        return response


def make_llm(settings, model: str | None = None) -> CachedLLM:
    client = GeminiClient(settings.gemini_api_key, model or settings.gemini_model)
    return CachedLLM(client, settings.data_dir / "cache" / "llm.sqlite")
