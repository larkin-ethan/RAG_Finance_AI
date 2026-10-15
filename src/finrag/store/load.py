"""CLI: embed a chunking strategy's chunks and load them into Postgres.

Usage:
  python -m finrag.store.load --strategy section-512             # everything
  python -m finrag.store.load --strategy section-512 --limit 1   # 1 filing: cheap smoke test
  python -m finrag.store.load --strategy fixed-512-64 --tickers AAPL MSFT

Re-runs are cheap: embeddings come from the disk cache, rows are upserted.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from itertools import islice

import psycopg
from pgvector import HalfVector

from finrag.embed.embedder import Embedder
from finrag.settings import get_settings
from finrag.store.db import connect

FILING_SQL = """
INSERT INTO filings VALUES (%(accession_number)s, %(ticker)s, %(company)s, %(form)s,
    %(filing_date)s, %(period_end)s, %(fiscal_year)s, %(fiscal_quarter)s, %(url)s)
ON CONFLICT (accession_number) DO NOTHING
"""
CHUNK_SQL = """
INSERT INTO chunks (chunk_id, embed_model, strategy, accession_number, ticker, form,
    fiscal_year, fiscal_quarter, section_key, section_title, text, embed_text, n_tokens, embedding)
VALUES (%(chunk_id)s, %(embed_model)s, %(strategy)s, %(accession_number)s, %(ticker)s, %(form)s,
    %(fiscal_year)s, %(fiscal_quarter)s, %(section_key)s, %(section_title)s, %(text)s,
    %(embed_text)s, %(n_tokens)s, %(embedding)s)
ON CONFLICT (chunk_id, embed_model) DO UPDATE SET embedding = EXCLUDED.embedding,
    text = EXCLUDED.text, embed_text = EXCLUDED.embed_text
"""
BATCH = 256


def read_chunks(path, tickers: set[str] | None, limit: int | None) -> list[dict]:
    chunks, filings = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if tickers and c["ticker"] not in tickers:
                continue
            if c["accession_number"] not in filings:
                if limit and len(filings) == limit:
                    continue
                filings.append(c["accession_number"])
            chunks.append(c)
    return chunks


def upsert_chunks(
    conn: psycopg.Connection,
    embedder: Embedder,
    chunks: list[dict],
    on_batch: Callable[[int, int], None] | None = None,
) -> None:
    """Embed chunk dicts (cache-first) and upsert them with their filings.
    Each batch commits on its own, so an interrupted load keeps its progress."""
    model = embedder.client.model
    done, it = 0, iter(chunks)
    while batch := list(islice(it, BATCH)):
        vectors = embedder.embed([c["embed_text"] for c in batch])
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(FILING_SQL, batch)
            cur.executemany(
                CHUNK_SQL,
                [
                    c | {"embed_model": model, "embedding": HalfVector(v)}
                    for c, v in zip(batch, vectors)
                ],
            )
        done += len(batch)
        if on_batch:
            on_batch(done, len(chunks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, help="e.g. section-512, fixed-512-64")
    parser.add_argument("--model", help="Voyage model (default: settings.voyage_model)")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--limit", type=int, help="max number of filings (smoke tests)")
    args = parser.parse_args()

    settings = get_settings()
    path = settings.data_dir / "chunks" / f"{args.strategy}.jsonl"
    chunks = read_chunks(path, set(args.tickers or []) or None, args.limit)
    embedder = Embedder.from_settings(settings, args.model)
    model = embedder.client.model
    print(f"{len(chunks):,} chunks from {path.name} -> {model}", flush=True)

    conn = connect(settings.database_url, init=True)
    started = time.time()

    def progress(done: int, total: int) -> None:
        s = embedder.stats
        print(f"  {done:,}/{total:,} (cached {s.cached:,})", end="\r", flush=True)

    upsert_chunks(conn, embedder, chunks, on_batch=progress)

    s, elapsed = embedder.stats, time.time() - started
    print(f"\nDone in {elapsed:.0f}s: {s.embedded:,} embedded, {s.cached:,} from cache")
    if s.embedded:
        rate = s.embedded / elapsed
        print(f"Throughput: {rate:.1f} chunks/s (~{s.approx_tokens / elapsed:,.0f} tokens/s)")
    if s.billed_tokens:
        ratio = s.billed_tokens / s.approx_tokens
        print(
            f"Billed {s.billed_tokens:,} tokens (estimate {s.approx_tokens:,}, ratio {ratio:.2f})"
        )
    if embedder.budget:
        print(
            f"Lifetime {model} spend: {embedder.budget.spent:,} / {embedder.budget.limit:,} tokens"
        )


if __name__ == "__main__":
    main()
