"""One configurable retrieval pipeline, used by the API and the eval runner.

  question -> [filters from question] -> vector and/or keyword search
           -> [RRF fusion] -> [cross-encoder rerank] -> top k

Each bracketed step is an experiment switch (RetrievalConfig), so the eval
can measure what each one buys.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from finrag.retrieve.filters import Filters
from finrag.retrieve.hybrid import rrf
from finrag.retrieve.query_parse import QueryParser
from finrag.retrieve.rerank import Reranker
from finrag.retrieve.vector import Hit

SearchFn = Callable[[str, int, Filters], list[Hit]]


@dataclass
class RetrievalConfig:
    strategy: str = "section-512"
    method: Literal["vector", "keyword", "hybrid"] = "vector"
    filters: Literal["none", "auto"] = "none"
    rerank: bool = False
    candidates: int = 30  # pool size per method before fusion / reranking


@dataclass
class Retriever:
    config: RetrievalConfig
    vector_fn: SearchFn | None = None
    keyword_fn: SearchFn | None = None
    parser: QueryParser | None = None
    reranker: Reranker | None = None
    last_filters: Filters = field(default_factory=Filters)  # for eval diagnostics

    def __call__(self, question: str, k: int) -> list[Hit]:
        cfg = self.config
        filters = Filters()
        if cfg.filters == "auto":
            if self.parser is None:
                raise ValueError("filters='auto' needs a QueryParser")
            filters = self.parser.parse(question)
        self.last_filters = filters

        pool = max(k, cfg.candidates) if (cfg.rerank or cfg.method == "hybrid") else k
        lists = []
        if cfg.method in ("vector", "hybrid"):
            lists.append(self._need(self.vector_fn, "vector_fn")(question, pool, filters))
        if cfg.method in ("keyword", "hybrid"):
            lists.append(self._need(self.keyword_fn, "keyword_fn")(question, pool, filters))
        hits = rrf(lists, limit=pool) if cfg.method == "hybrid" else lists[0]

        if cfg.rerank:
            hits = self._need(self.reranker, "reranker").rerank(question, hits, k)
        return hits[:k]

    @staticmethod
    def _need(obj, name: str):
        if obj is None:
            raise ValueError(f"this RetrievalConfig requires {name}")
        return obj


def db_retriever(conn, embedder, config: RetrievalConfig, tickers: list[str]) -> Retriever:
    """Wire a Retriever to Postgres + an Embedder."""
    from finrag.retrieve.keyword import keyword_search
    from finrag.retrieve.vector import vector_search

    model = embedder.client.model

    def vector_fn(q: str, n: int, f: Filters) -> list[Hit]:
        return vector_search(conn, embedder.embed_query(q), config.strategy, model, n, f)

    def keyword_fn(q: str, n: int, f: Filters) -> list[Hit]:
        return keyword_search(conn, q, config.strategy, model, n, f)

    return Retriever(
        config,
        vector_fn=vector_fn,
        keyword_fn=keyword_fn,
        parser=QueryParser.for_tickers(tickers),
        reranker=Reranker() if config.rerank else None,
    )
