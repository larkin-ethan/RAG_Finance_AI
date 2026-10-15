"""Disk cache of embeddings, so the same text is never paid for twice.

Keyed by sha256(model | dim | input_type | text). Re-running ingestion, or
re-loading a strategy into a fresh database, costs zero API tokens.
Stored in SQLite as float32 bytes (4 KB per 1024-dim vector).
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array
from pathlib import Path


def cache_key(model: str, dim: int | None, input_type: str, text: str) -> str:
    raw = f"{model}|{dim}|{input_type}|{text}".encode()
    return hashlib.sha256(raw).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):  # stay under SQLite's variable limit
            part = keys[i : i + 500]
            marks = ",".join("?" * len(part))
            for key, blob in self._db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({marks})", part
            ):
                found[key] = array("f", blob).tolist()
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            [(k, array("f", v).tobytes()) for k, v in items.items()],
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
