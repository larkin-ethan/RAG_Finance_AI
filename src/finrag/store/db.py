"""Postgres connection helpers."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

SCHEMA = Path(__file__).with_name("schema.sql")


def auto_migrate_enabled() -> bool:
    """Dev default: create/upgrade the schema on connect. The private k8s
    deployment sets DB_AUTO_MIGRATE=false: there the app connects as a role
    without DDL rights, and a separate migrate Job (finrag.store.migrate)
    applies the schema as the owner role."""
    return os.environ.get("DB_AUTO_MIGRATE", "true").lower() not in ("0", "false", "no")


def connect(database_url: str, init: bool = False) -> psycopg.Connection:
    """Open a connection. With init=True (and auto-migrate on), create the schema
    first (idempotent). The vector extension must exist before pgvector types
    can be registered."""
    # Without a timeout, an unreachable database hangs the caller forever.
    conn = psycopg.connect(database_url, autocommit=True, connect_timeout=5)
    if init and auto_migrate_enabled():
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
    register_vector(conn)
    return conn
