"""Local embedding model via sentence-transformers. Free, private, no rate limits.

Default: Qwen3-Embedding-0.6B (Apache-2.0, 1024 dims, 32K context). Same
dimension as voyage-4, so both share the chunks.embedding halfvec(1024) column.

Asymmetric encoding, per the model card: queries get a one-line task
instruction, documents get none. Skipping the instruction costs ~1-5% in
retrieval quality. Ours is specific to SEC filings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from finrag.embed.voyage_client import EmbedResult

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Instruct: Given a question about a public company, retrieve passages from its "
    "SEC 10-K and 10-Q filings that answer the question\nQuery:"
)
TEXTS_PER_BATCH = 256  # cache checkpoint granularity (one SQLite write per batch)
# GPU batch size per device: small for 8 GB Macs (MPS), large for a 24 GB CUDA GPU.
ENCODE_BATCH_SIZE = {"mps": 8, "cuda": 64, "cpu": 16}
MAX_SEQ_LENGTH = 1024  # our chunks are <= ~512 tokens; caps memory on outliers


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cuda" if torch.cuda.is_available() else "cpu"


class LocalClient:
    """Same interface as VoyageClient (model, dim, batches, embed_batch, cache_tag)."""

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, dim: int = 1024, encoder: Any = None):
        self.model, self.dim = model, dim
        self.device = "cpu"
        self._encoder = encoder  # tests inject a fake; otherwise loaded lazily

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.device = _device()
            # fp16 halves memory and is ~2x faster on GPU; CPU needs fp32.
            dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
            self._encoder = SentenceTransformer(
                self.model,
                device=device,
                model_kwargs={"torch_dtype": dtype},
                truncate_dim=self.dim,
            )
            self._encoder.max_seq_length = MAX_SEQ_LENGTH
        return self._encoder

    def cache_tag(self, input_type: str) -> str:
        """Part of the cache key. Query vectors depend on the instruction text,
        so changing it must not reuse stale cached query vectors."""
        if input_type == "query":
            return "query:" + hashlib.sha256(QUERY_INSTRUCTION.encode()).hexdigest()[:8]
        return input_type

    def batches(self, texts: list[str]) -> Iterator[list[str]]:
        for i in range(0, len(texts), TEXTS_PER_BATCH):
            yield texts[i : i + TEXTS_PER_BATCH]

    def embed_batch(self, texts: list[str], input_type: str) -> EmbedResult:
        vectors = self.encoder.encode(
            texts,
            prompt=QUERY_INSTRUCTION if input_type == "query" else None,
            batch_size=ENCODE_BATCH_SIZE.get(self.device, 8),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if np.isnan(vectors).any():  # fp16 overflow would show up as NaN
            raise RuntimeError("NaN in embeddings; try float32 (CPU) instead of fp16")
        return EmbedResult(vectors.tolist(), tokens=0)  # local models aren't billed
