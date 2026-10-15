"""Cache-first embedding: look up every text, send only the misses to the
backend (local model, or Voyage within its token budget), store the results."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from finrag.chunking.base import count_tokens
from finrag.embed.budget import TokenBudget
from finrag.embed.cache import EmbeddingCache, cache_key
from finrag.embed.voyage_client import EmbedResult, VoyageClient
from finrag.settings import Settings


class EmbeddingBackend(Protocol):
    """What Embedder needs from a backend: VoyageClient and LocalClient both fit."""

    model: str
    dim: int | None

    def cache_tag(self, input_type: str) -> str: ...
    def batches(self, texts: list[str]) -> Iterator[list[str]]: ...
    def embed_batch(self, texts: list[str], input_type: str) -> EmbedResult: ...


@dataclass
class EmbedStats:
    cached: int = 0
    embedded: int = 0
    billed_tokens: int = 0
    approx_tokens: int = 0  # our estimate for the same texts (calibration, throughput)
    batches: list[int] = field(default_factory=list)


class Embedder:
    def __init__(self, client: EmbeddingBackend, cache: EmbeddingCache, budget: TokenBudget | None):
        self.client, self.cache, self.budget = client, cache, budget
        self.stats = EmbedStats()

    @classmethod
    def from_settings(cls, settings: Settings, model: str | None = None) -> Embedder:
        cache = EmbeddingCache(settings.data_dir / "cache" / "embeddings.sqlite")
        if settings.embed_provider == "local":
            from finrag.embed.local_client import LocalClient  # imports torch: keep lazy

            client = LocalClient(model or settings.local_embed_model, settings.embed_dim)
            return cls(client, cache, budget=None)  # nothing to bill

        model = model or settings.voyage_model
        client = VoyageClient(
            settings.voyage_api_key,
            model,
            settings.embed_dim,
            max_batch_tokens=settings.voyage_max_batch_tokens,
            requests_per_minute=settings.voyage_rpm,
        )
        budget = TokenBudget(
            settings.data_dir / "cache" / "token_ledger.json", settings.embed_budget_tokens, model
        )
        return cls(client, cache, budget)

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        c = self.client
        tag = c.cache_tag(input_type)
        keys = [cache_key(c.model, c.dim, tag, t) for t in texts]
        found = self.cache.get_many(list(dict.fromkeys(keys)))
        self.stats.cached += sum(k in found for k in keys)

        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
        for batch in c.batches(missing):
            approx = sum(count_tokens(t) for t in batch)
            if self.budget:
                self.budget.check(approx)
            result = c.embed_batch(batch, input_type)
            if self.budget:
                self.budget.record(result.tokens)
            new = {cache_key(c.model, c.dim, tag, t): v for t, v in zip(batch, result.vectors)}
            self.cache.put_many(new)  # checkpoint: an interrupted load resumes from here
            found.update(new)
            self.stats.embedded += len(batch)
            self.stats.billed_tokens += result.tokens
            self.stats.approx_tokens += approx
            self.stats.batches.append(len(batch))
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], input_type="query")[0]
