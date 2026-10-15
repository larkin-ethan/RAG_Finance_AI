"""Daily job: pull any new 10-K/10-Q filings for every watchlist company.

  python -m finrag.jobs.update_filings

On Google Cloud this runs as a Cloud Run Job triggered daily by Cloud Scheduler.
Locally, the API can run it on startup (AUTO_UPDATE_ON_START=true) if the last
successful run is more than 24h old.

Cheap when nothing changed: each company is checked only for filings after
the newest one already loaded, and every pipeline step is idempotent. Exits
non-zero if any company failed, so the scheduler can alert and retry.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import psycopg

from finrag.embed.embedder import Embedder
from finrag.pipeline import ingest_ticker
from finrag.settings import Settings, get_settings, load_watchlist
from finrag.store.db import connect

JOB = "update_filings"
OVERLAP = timedelta(days=7)  # re-check a week back: amended dates, late index updates
STALE_AFTER = timedelta(hours=24)


def latest_filing_date(conn: psycopg.Connection, ticker: str) -> date | None:
    row = conn.execute(
        "SELECT MAX(filing_date) FROM filings WHERE ticker = %s", (ticker,)
    ).fetchone()
    return row[0] if row else None


def last_success_age(conn: psycopg.Connection) -> timedelta | None:
    row = conn.execute(
        "SELECT now() - MAX(finished_at) FROM job_runs WHERE job = %s AND ok", (JOB,)
    ).fetchone()
    return row[0] if row else None


def run(settings: Settings, conn: psycopg.Connection, embedder) -> dict:
    watchlist = load_watchlist(settings.watchlist_path)
    run_id = conn.execute("INSERT INTO job_runs (job) VALUES (%s) RETURNING id", (JOB,)).fetchone()[
        0
    ]
    results: dict[str, dict] = {}
    for ticker in watchlist.tickers:
        last = latest_filing_date(conn, ticker)
        since = (last - OVERLAP) if last else watchlist.since
        try:
            n = ingest_ticker(
                settings, conn, embedder, ticker,
                forms=tuple(watchlist.forms), since=since,
                extra_ciks=tuple(watchlist.extra_ciks.get(ticker, [])),
            )  # fmt: skip
            results[ticker] = {"ok": True, "since": str(since), "chunks": n}
        except Exception as exc:  # noqa: BLE001 - one bad ticker mustn't stop the rest
            results[ticker] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"{ticker:6} {results[ticker]}", flush=True)
    ok = all(r["ok"] for r in results.values())
    conn.execute(
        "UPDATE job_runs SET finished_at = now(), ok = %s, summary = %s WHERE id = %s",
        (ok, json.dumps(results), run_id),
    )
    return {"ok": ok, "results": results}


def main() -> None:
    settings = get_settings()
    conn = connect(settings.database_url, init=True)
    outcome = run(settings, conn, Embedder.from_settings(settings))
    sys.exit(0 if outcome["ok"] else 1)


if __name__ == "__main__":
    main()
