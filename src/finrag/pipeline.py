"""End-to-end ingestion for one ticker: EDGAR -> parse -> chunk -> embed -> Postgres.

Used by the API's "Load Filings" page (with progress callbacks) and by the
daily new-filing job. Every step is idempotent: already-downloaded filings
are skipped, cached embeddings cost nothing, and rows are upserted. Re-running
for a ticker only does work for filings that are new.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date

import psycopg

from finrag.chunking.base import Filing
from finrag.chunking.section_aware import chunk_section_aware
from finrag.embed.embedder import Embedder
from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.fetch import save_filing
from finrag.ingest.parse import parse_filing
from finrag.settings import Settings
from finrag.store.load import upsert_chunks

APP_STRATEGY_TOKENS = 512  # the strategy the app serves (see retrieval default)


@dataclass
class Progress:
    stage: str  # "listing" | "downloading" | "parsing" | "embedding" | "done" | "error"
    done: int = 0
    total: int = 0
    message: str = ""


def ingest_ticker(
    settings: Settings,
    conn: psycopg.Connection,
    embedder: Embedder,
    ticker: str,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    since: date | None = None,
    extra_ciks: tuple[str, ...] = (),
    on_progress: Callable[[Progress], None] = lambda p: None,
) -> int:
    """Ingest a ticker's filings. Returns the number of chunks upserted."""
    ticker = ticker.upper()
    client = EdgarClient(settings.sec_user_agent)
    raw_dir, parsed_dir = settings.data_dir / "raw", settings.data_dir / "parsed"

    on_progress(Progress("listing", message=f"Looking up {ticker} on EDGAR"))
    info = client.company_info(ticker)
    refs = client.list_filings(ticker, forms, since, extra_ciks)

    meta_files = []
    for i, ref in enumerate(refs):
        on_progress(Progress("downloading", i, len(refs), f"{ref.form} filed {ref.filing_date}"))
        meta_files.append(save_filing(client, ref, info, raw_dir))

    chunks: list[dict] = []
    for i, meta_file in enumerate(meta_files):
        on_progress(Progress("parsing", i, len(meta_files), meta_file.name))
        parse_filing(meta_file, parsed_dir)
        filing = Filing.load(parsed_dir / ticker / meta_file.name.replace(".meta", ""))
        chunks += [asdict(c) for c in chunk_section_aware(filing, APP_STRATEGY_TOKENS)]

    def batch_done(done: int, total: int) -> None:
        on_progress(Progress("embedding", done, total, f"{len(refs)} filings"))

    upsert_chunks(conn, embedder, chunks, on_batch=batch_done)
    on_progress(Progress("done", len(chunks), len(chunks), f"{info.name}: {len(refs)} filings"))
    return len(chunks)
