"""Embeddings served by a local Ollama (GPU) via /api/embed.

Used by the private local deployment: the app containers then need no PyTorch
(smaller images, fewer dependencies to patch), and nothing leaves the machine.

Vectors from Ollama's quantized model differ slightly from sentence-transformers'
for the same model, so they're stored under a distinct model label
("ollama/qwen3-embedding:0.6b"). Corpus and queries always come from one backend.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator

import httpx

from finrag.embed.local_client import QUERY_INSTRUCTION
from finrag.embed.voyage_client import EmbedResult

EMBED_PATH = "/api/embed"
TEXTS_PER_REQUEST = 64


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class OllamaEmbedClient:
    """Same interface as LocalClient / VoyageClient (model, dim, cache_tag, batches, embed_batch)."""

    def __init__(
        self,
        base_url: str,
        model: str = "qwen3-embedding:0.6b",
        dim: int = 1024,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 3,
    ):
        self.ollama_model, self.dim, self.max_attempts = model, dim, max_attempts
        self.model = f"ollama/{model}"  # label stored with vectors; never mixed with other backends
        self._http = httpx.Client(
            base_url=base_url, timeout=300.0, transport=transport, trust_env=False
        )

    def cache_tag(self, input_type: str) -> str:
        if input_type == "query":
            return "query:" + hashlib.sha256(QUERY_INSTRUCTION.encode()).hexdigest()[:8]
        return input_type

    def batches(self, texts: list[str]) -> Iterator[list[str]]:
        for i in range(0, len(texts), TEXTS_PER_REQUEST):
            yield texts[i : i + TEXTS_PER_REQUEST]

    def embed_batch(self, texts: list[str], input_type: str) -> EmbedResult:
        # Qwen3-Embedding is asymmetric: queries carry a task instruction,
        # documents don't (same text the sentence-transformers backend uses).
        inputs = [QUERY_INSTRUCTION + t for t in texts] if input_type == "query" else texts
        body = {
            "model": self.ollama_model,
            "input": inputs,
            "dimensions": self.dim,
            "truncate": True,
        }
        for attempt in range(self.max_attempts):
            try:
                response = self._http.post(EMBED_PATH, json=body)
            except httpx.TransportError as exc:
                if attempt == self.max_attempts - 1:
                    raise RuntimeError(
                        f"Can't reach Ollama at {self._http.base_url}: {exc}"
                    ) from exc
                time.sleep(2 ** (attempt + 1))
                continue
            if response.is_error:
                raise RuntimeError(f"Ollama embed {response.status_code}: {response.text[:400]}")
            vectors = [_normalize(v) for v in response.json()["embeddings"]]
            if len(vectors) != len(texts) or any(len(v) != self.dim for v in vectors):
                raise RuntimeError(
                    f"Ollama returned unexpected embeddings shape for {self.ollama_model}"
                )
            return EmbedResult(vectors, tokens=0)
        raise AssertionError("unreachable")
