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
OLLAMA_CHAT_PATH = "/api/chat"


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


def to_json_schema(schema: dict) -> dict:
    """Convert our Gemini-style schema (OBJECT/STRING types, propertyOrdering)
    to standard JSON Schema, which Ollama's structured outputs expect."""
    out = {}
    for key, value in schema.items():
        if key == "propertyOrdering":
            continue
        if key == "type" and isinstance(value, str):
            out[key] = value.lower()
        elif key == "properties":
            out[key] = {k: to_json_schema(v) for k, v in value.items()}
        elif key == "items":
            out[key] = to_json_schema(value)
        else:
            out[key] = value
    return out


class OllamaClient:
    """Local LLM via Ollama's /api/chat (runs on your GPU; nothing leaves the machine).

    num_ctx matters: Ollama's default context window is small and longer prompts
    are SILENTLY truncated, so the model would answer from half the evidence.
    Our prompts (6 passages x ~512 tokens + instructions) need ~5-6K tokens.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        transport: httpx.BaseTransport | None = None,
        num_ctx: int = 16384,
        max_attempts: int = 3,
        think: bool = False,  # disable "thinking" mode on reasoning models: faster, JSON-only
    ):
        self.model, self.num_ctx, self.max_attempts, self.think = (
            model,
            num_ctx,
            max_attempts,
            think,
        )
        # trust_env=False: local traffic must never be routed through HTTPS_PROXY.
        self._http = httpx.Client(
            base_url=base_url, timeout=300.0, transport=transport, trust_env=False
        )

    def generate(
        self, system: str, prompt: str, schema: dict | None = None, temperature: float = 0.0
    ) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": self.think,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if schema:
            json_schema = to_json_schema(schema)
            body["format"] = json_schema
            # Ollama recommends also stating the schema in the prompt to ground the output.
            body["messages"][1]["content"] += (
                "\n\nRespond with JSON matching this schema:\n" + json.dumps(json_schema)
            )
        started = time.time()
        data = self._post(body)
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=time.time() - started,
        )

    def _post(self, body: dict) -> dict:
        for attempt in range(self.max_attempts):
            try:
                response = self._http.post(OLLAMA_CHAT_PATH, json=body)
            except httpx.TransportError as exc:  # Ollama not running / restarting
                if attempt == self.max_attempts - 1:
                    raise LLMError(f"Can't reach Ollama at {self._http.base_url}: {exc}") from exc
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code >= 500 and attempt < self.max_attempts - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            if response.is_error:
                raise LLMError(f"Ollama {response.status_code}: {response.text[:600]}")
            return response.json()
        raise AssertionError("unreachable")


def make_llm(settings, role: str = "answer", provider: str | None = None) -> CachedLLM:
    """The configured LLM for a role ("answer" or "judge"), wrapped in the cache.
    The judge is deliberately a different model from the answerer."""
    provider = provider or settings.llm_provider
    if provider == "ollama":
        model = settings.ollama_judge_model if role == "judge" else settings.ollama_model
        client: LLMClient = OllamaClient(settings.ollama_url, model)
    else:
        model = settings.gemini_judge_model if role == "judge" else settings.gemini_model
        client = GeminiClient(settings.gemini_api_key, model)
    return CachedLLM(client, settings.data_dir / "cache" / "llm.sqlite")
