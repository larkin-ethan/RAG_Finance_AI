"""Postgres connection helpers."""

from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

SCHEMA = Path(__file__).with_name("schema.sql")


def connect(database_url: str, init: bool = False) -> psycopg.Connection:
    """Open a connection. With init=True, create the schema first (idempotent).
    The vector extension must exist before pgvector types can be registered."""
    # Without a timeout, an unreachable database hangs the caller forever.
    conn = psycopg.connect(database_url, autocommit=True, connect_timeout=5)
    if init:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
    register_vector(conn)
    return conn
