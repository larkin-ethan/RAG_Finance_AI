"""Cross-encoder reranking: score (question, passage) pairs jointly.

A bi-encoder (our embedding model) embeds question and passage separately; a
cross-encoder reads them together, which is slower but more precise. So we
rerank only the top candidates (e.g. 30) from vector/hybrid search.

Default: BAAI/bge-reranker-v2-m3 (Apache-2.0, ~570M params) via
sentence-transformers' CrossEncoder; runs on CUDA, Apple MPS, or CPU.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from finrag.retrieve.vector import Hit

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model: str = DEFAULT_RERANKER, scorer: Any = None, max_length: int = 1024):
        self.model, self.max_length = model, max_length
        self._scorer = scorer  # tests inject a fake

    @property
    def scorer(self) -> Any:
        if self._scorer is None:
            from sentence_transformers import CrossEncoder

            from finrag.embed.local_client import _device

            self._scorer = CrossEncoder(self.model, device=_device(), max_length=self.max_length)
        return self._scorer

    def rerank(self, question: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.scorer.predict([(question, h.text) for h in hits], batch_size=16)
        ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [replace(h, score=float(s)) for h, s in ranked[:top_k]]
