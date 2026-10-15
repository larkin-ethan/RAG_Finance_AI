"""FastAPI backend.

Run locally: uv run uvicorn finrag.api.main:app --reload
Docs:        http://localhost:8000/docs
"""

from __future__ import annotations

import hmac
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from finrag.api.ratelimit import RateLimiter, client_ip
from finrag.api.schemas import AskRequest, FeedbackRequest, IngestRequest
from finrag.api.services import Services
from finrag.settings import get_settings

DEFAULT_LOOKBACK = timedelta(days=3 * 365)
RESULTS_DIR = Path("evals/results")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.auto_update_on_start:
        threading.Thread(target=_catch_up, args=(app,), daemon=True, name="catch-up").start()
    yield


def _catch_up(app: FastAPI) -> None:
    """Locally: check EDGAR for new filings if the last check is >24h old."""
    from finrag.jobs.update_filings import STALE_AFTER, last_success_age, run
    from finrag.store.db import connect

    svc = app.state.services = getattr(app.state, "services", None) or Services(get_settings())
    try:
        conn = connect(svc.settings.database_url, init=True)
        age = last_success_age(conn)
        if age is None or age > STALE_AFTER:
            run(svc.settings, conn, svc.embedder)
    except Exception as exc:  # noqa: BLE001 - never crash the API over a background refresh
        print(f"catch-up update failed: {exc}", flush=True)


app = FastAPI(title="FinRAG", description="Cited Q&A over SEC filings", lifespan=lifespan)


def services(request: Request) -> Services:
    if not hasattr(request.app.state, "services"):
        request.app.state.services = Services(get_settings())
    return request.app.state.services


def limiter(request: Request) -> RateLimiter:
    if not hasattr(request.app.state, "limiter"):
        s = services(request).settings
        request.app.state.limiter = RateLimiter(s.ask_per_minute, s.daily_question_cap)
    return request.app.state.limiter


def is_admin(request: Request, token: str) -> bool:
    expected = services(request).settings.admin_token
    # Constant-time compare. No token configured means nobody is admin.
    return bool(expected) and hmac.compare_digest(token, expected)


def caller(request: Request) -> str:
    s = services(request).settings
    peer = request.client.host if request.client else None
    return client_ip(request.headers, peer, s.trust_proxy)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "db": services(request).db_ok()}


@app.post("/ask")
def ask(body: AskRequest, request: Request) -> dict:
    public = services(request).settings.public_demo
    if public and (reason := limiter(request).check(caller(request))):
        raise HTTPException(429, reason)
    return asdict(services(request).answer(body))


@app.get("/companies")
def companies(request: Request) -> list[dict]:
    return services(request).companies()


@app.get("/filings")
def filings(
    request: Request, ticker: str | None = None, form: str | None = None, year: int | None = None
) -> list[dict]:
    return services(request).filings(ticker, form, year)


@app.get("/filings/{accession}")
def filing_detail(accession: str, request: Request) -> dict:
    detail = services(request).filing_detail(accession)
    if detail is None:
        raise HTTPException(404, f"Filing {accession} not found")
    return detail


@app.post("/ingest")
def ingest(body: IngestRequest, request: Request, x_admin_token: str = Header(default="")) -> dict:
    # Ingestion downloads and embeds whole filing histories. Not for anonymous users.
    if services(request).settings.public_demo and not is_admin(request, x_admin_token):
        raise HTTPException(403, "Loading new companies is disabled in the public demo")
    since = body.since or (date.today() - DEFAULT_LOOKBACK)
    return {"job_id": services(request).start_ingest(body.ticker, since)}


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str, request: Request) -> dict:
    job = services(request).jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


@app.post("/feedback")
def feedback(body: FeedbackRequest, request: Request) -> dict:
    return {"id": services(request).save_feedback(body, caller(request))}


@app.get("/feedback")
def list_feedback(
    request: Request, limit: int = 100, x_admin_token: str = Header(default="")
) -> list[dict]:
    if not is_admin(request, x_admin_token):
        raise HTTPException(403, "Admin token required")
    return services(request).list_feedback(min(limit, 500))


@app.get("/evals")
def evals() -> list[dict]:
    """Saved eval runs (summaries only), newest first, for the dashboard."""
    runs = []
    for path in sorted(RESULTS_DIR.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
        except (json.JSONDecodeError, KeyError):
            continue
        runs.append({"file": str(path.relative_to(RESULTS_DIR)), "summary": summary,
                     "modified": path.stat().st_mtime})  # fmt: skip
    return runs


@app.get("/config")
def config(request: Request) -> dict:
    """Non-secret settings, for the read-only Settings page."""
    s = services(request).settings
    return {
        "embed_provider": s.embed_provider,
        "embed_model": s.local_embed_model if s.embed_provider == "local" else s.voyage_model,
        "embed_dim": s.embed_dim,
        "llm_model": s.gemini_model,
        "judge_model": s.gemini_judge_model,
        "chunk_strategy": "section-512",
        "watchlist": s.watchlist_path.name,
    }
