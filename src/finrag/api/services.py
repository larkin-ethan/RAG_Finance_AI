"""Everything the API needs (DB, embedder, LLM, ingestion jobs) behind one object.
Routes stay thin, and tests swap in a fake Services."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import asdict
from datetime import date
from functools import cached_property
from typing import Any

from finrag.api.schemas import AskRequest, FeedbackRequest
from finrag.embed.embedder import Embedder
from finrag.generate.answer import Answer, answer_question
from finrag.generate.llm import make_llm
from finrag.pipeline import Progress, ingest_ticker
from finrag.retrieve.filters import Filters
from finrag.retrieve.vector import Hit, vector_search
from finrag.settings import Settings
from finrag.store.db import connect

APP_STRATEGY = "section-512"


class _LockedEmbedder:
    """Serializes access to the (GPU) embedding model across threads."""

    def __init__(self, inner: Embedder, lock: threading.Lock):
        self._inner, self._lock = inner, lock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        with self._lock:
            return self._inner.embed(texts, input_type)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], "query")[0]


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._db_lock, self._embed_lock = threading.Lock(), threading.Lock()
        self.jobs: dict[str, dict] = {}

    # Lazy: the API starts (and /health works) even before the model is loaded.
    @cached_property
    def conn(self):
        return connect(self.settings.database_url, init=True)

    @cached_property
    def embedder(self) -> _LockedEmbedder:
        return _LockedEmbedder(Embedder.from_settings(self.settings), self._embed_lock)

    @cached_property
    def llm(self):
        return make_llm(self.settings)

    @property
    def embed_model(self) -> str:
        return self.embedder.client.model

    def db_ok(self) -> bool:
        try:
            with self._db_lock:
                self.conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - health check reports any failure as "down"
            return False

    def retrieve(self, question: str, filters: Filters, k: int) -> list[Hit]:
        vector = self.embedder.embed_query(question)
        with self._db_lock:
            return vector_search(self.conn, vector, APP_STRATEGY, self.embed_model, k, filters)

    def answer(self, req: AskRequest) -> Answer:
        filters = Filters(
            tickers=req.tickers or None,
            fiscal_years=req.fiscal_years or None,
            forms=req.forms or None,
        )
        return answer_question(req.question, lambda q: self.retrieve(q, filters, req.k), self.llm)

    def _query(self, sql: str, params: dict | None = None) -> list[dict]:
        with self._db_lock, self.conn.cursor() as cur:
            cur.execute(sql, params or {})
            names = [d.name for d in cur.description]
            return [dict(zip(names, row)) for row in cur.fetchall()]

    def companies(self) -> list[dict]:
        return self._query("""
            SELECT ticker, company, COUNT(*) AS filings,
                   MIN(period_end)::text AS first_period, MAX(period_end)::text AS last_period
            FROM filings GROUP BY ticker, company ORDER BY ticker
        """)  # fmt: skip

    def filings(self, ticker: str | None, form: str | None, year: int | None) -> list[dict]:
        return self._query(
            """
            SELECT accession_number, ticker, company, form, filing_date::text,
                   period_end::text, fiscal_year, fiscal_quarter, url
            FROM filings
            WHERE (%(ticker)s::text IS NULL OR ticker = %(ticker)s)
              AND (%(form)s::text IS NULL OR form = %(form)s)
              AND (%(year)s::int IS NULL OR fiscal_year = %(year)s)
            ORDER BY filing_date DESC
            """,
            {"ticker": ticker and ticker.upper(), "form": form and form.upper(), "year": year},
        )

    def filing_detail(self, accession: str) -> dict | None:
        meta = self._query(
            "SELECT *, filing_date::text AS filing_date, period_end::text AS period_end "
            "FROM filings WHERE accession_number = %(a)s",
            {"a": accession},
        )
        if not meta:
            return None
        rows = self._query(
            """
            SELECT chunk_id, section_key, section_title, text FROM chunks
            WHERE accession_number = %(a)s AND strategy = %(s)s AND embed_model = %(m)s
            ORDER BY chunk_id
            """,
            {"a": accession, "s": APP_STRATEGY, "m": self.embed_model},
        )
        sections: list[dict] = []
        for r in rows:
            if not sections or sections[-1]["key"] != r["section_key"]:
                sections.append(
                    {"key": r["section_key"], "title": r["section_title"], "chunks": []}
                )
            sections[-1]["chunks"].append({"chunk_id": r["chunk_id"], "text": r["text"]})
        return meta[0] | {"sections": sections}

    def start_ingest(self, ticker: str, since: date | None) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {"ticker": ticker.upper()} | asdict(Progress("queued"))

        def update(p: Progress) -> None:
            self.jobs[job_id] |= asdict(p)

        def run() -> None:
            try:
                conn = connect(self.settings.database_url, init=True)  # own connection
                ingest_ticker(
                    self.settings, conn, self.embedder, ticker, since=since, on_progress=update
                )
            except Exception as exc:  # noqa: BLE001 - thread must report every failure to the UI
                update(Progress("error", message=f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=run, daemon=True, name=f"ingest-{ticker}").start()
        return job_id

    def save_feedback(self, fb: FeedbackRequest, client_ip: str) -> int:
        salt = self.settings.admin_token or "finrag"
        client_hash = hashlib.sha256(f"{salt}:{client_ip}".encode()).hexdigest()[:16]
        with self._db_lock:
            row = self.conn.execute(
                """INSERT INTO feedback (rating, issue_type, question, answer, details,
                                         citations, model, client_hash)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (fb.rating, fb.issue_type, fb.question, fb.answer, fb.details,
                 json.dumps(fb.citations), fb.model, client_hash),
            ).fetchone()  # fmt: skip
        return row[0]

    def list_feedback(self, limit: int = 100) -> list[dict]:
        return self._query(
            """SELECT id, created_at::text, rating, issue_type, question, answer, details,
                      citations, model FROM feedback ORDER BY created_at DESC LIMIT %(n)s""",
            {"n": limit},
        )
