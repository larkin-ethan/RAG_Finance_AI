"""Tiny HTTP client for the FinRAG API. The UI never imports backend code,
so it can be deployed as its own service (e.g. a separate Cloud Run service)."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("FINRAG_API_URL", "http://localhost:8000")
_TOKEN = os.environ.get("FINRAG_API_TOKEN", "")  # service token (private deployment)
_client = httpx.Client(
    base_url=API_URL,
    timeout=180.0,
    headers={"Authorization": f"Bearer {_TOKEN}"} if _TOKEN else {},
)


def get(path: str, **params) -> dict | list:
    r = _client.get(path, params={k: v for k, v in params.items() if v is not None})
    r.raise_for_status()
    return r.json()


def post(path: str, payload: dict) -> dict:
    r = _client.post(path, json=payload)
    r.raise_for_status()
    return r.json()


def health() -> dict:
    try:
        return get("/health")
    except httpx.HTTPError:
        return {"status": "down", "db": False}


@st.cache_data(ttl=30)
def companies() -> list[dict]:
    try:
        return get("/companies")
    except httpx.HTTPError:
        return []


def sidebar_status() -> None:
    """Called at the top of every page: enforces login first (if configured)."""
    from auth import require_login

    require_login()
    h = health()
    ok = h.get("status") == "ok" and h.get("db")
    st.sidebar.markdown(f"{'🟢' if ok else '🔴'} System {'online' if ok else 'offline'}")
    st.sidebar.caption("Powered by Qwen3 embeddings · Gemini · pgvector")


def put(path: str, payload: dict) -> dict:
    r = _client.put(path, json=payload)
    r.raise_for_status()
    return r.json()


def delete(path: str) -> dict:
    r = _client.delete(path)
    r.raise_for_status()
    return r.json()


DISCLAIMER = (
    "⚠️ **Research tool, not investment or tax advice.** Grades describe how well a stock's "
    "reported data fits a goal; they don't predict returns. Tax figures are federal estimates."
)


def require_ratings() -> None:
    """Stop the page unless this deployment has stock checks enabled (private only)."""
    try:
        enabled = get("/config").get("ratings_enabled")
    except httpx.HTTPError:
        enabled = False
    if not enabled:
        st.info(
            "Stock checks and portfolio features are only available in the private local deployment."
        )
        st.stop()
    st.warning(DISCLAIMER)


def get_or_stop(path: str, **params):
    """GET, or show a friendly error and stop the page (no tracebacks in the browser)."""
    try:
        return get(path, **params)
    except httpx.HTTPStatusError as exc:
        detail = (
            exc.response.json().get("detail")
            if exc.response.headers.get("content-type", "").startswith("application/json")
            else None
        )
        st.error(detail or f"The API returned an error ({exc.response.status_code}).")
    except httpx.HTTPError:
        st.error("Can't reach the API or the database. Is everything running?")
    st.stop()
