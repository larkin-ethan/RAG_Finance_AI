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
