"""CLI: embed a strategy's chunks into the disk cache only (no database needed).

Use this on a machine with a strong GPU, then copy the cache file
(data/cache/embeddings.sqlite) to the machine that runs Postgres. There,
`finrag.store.load` finds every vector in the cache and computes nothing.
Cache keys are (model, dim, input type, text), so the same chunks produce
the same keys on any OS.

Usage: python -m finrag.embed.run --strategy section-512 [--limit N]
"""

from __future__ import annotations

import argparse
import time

from finrag.embed.embedder import Embedder
from finrag.settings import get_settings
from finrag.store.load import read_chunks

PROGRESS_EVERY = 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--limit", type=int, help="max number of filings")
    args = parser.parse_args()

    settings = get_settings()
    path = settings.data_dir / "chunks" / f"{args.strategy}.jsonl"
    texts = [
        c["embed_text"] for c in read_chunks(path, set(args.tickers or []) or None, args.limit)
    ]
    embedder = Embedder.from_settings(settings)
    print(f"{len(texts):,} chunks -> {embedder.client.model}", flush=True)

    started = time.time()
    for i in range(0, len(texts), PROGRESS_EVERY):
        embedder.embed(texts[i : i + PROGRESS_EVERY])
        done, elapsed = min(i + PROGRESS_EVERY, len(texts)), time.time() - started
        print(
            f"  {done:,}/{len(texts):,}  {embedder.stats.embedded / elapsed:.1f} chunks/s",
            flush=True,
        )
    s = embedder.stats
    print(f"Done in {time.time() - started:.0f}s: {s.embedded:,} embedded, {s.cached:,} cached")
    print(f"Cache: {embedder.cache.path} ({len(embedder.cache):,} vectors)")


if __name__ == "__main__":
    main()
