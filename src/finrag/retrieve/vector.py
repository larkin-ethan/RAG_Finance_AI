"""Dense (vector) retrieval with metadata filters.

Filtered approximate search pitfall: an HNSW index scan finds the ~ef_search
nearest vectors first and applies the WHERE filter afterwards. Ask for AAPL
chunks in a table that is 90% other companies, and you can get back far fewer
than k rows. pgvector 0.8's iterative scan keeps walking the graph until k
rows pass the filter, so it's turned on for every search.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector import HalfVector

from finrag.retrieve.filters import Filters


@dataclass
class Hit:
    chunk_id: str
    score: float  # cosine similarity, higher is better
    ticker: str
    company: str
    form: str
    fiscal_year: int
    fiscal_quarter: int | None
    period_end: str
    section_key: str
    section_title: str
    url: str
    text: str


def build_search_sql(filters: Filters) -> tuple[str, dict]:
    clauses, params = filters.where()
    where = " AND ".join(["c.strategy = %(strategy)s", "c.embed_model = %(model)s", *clauses])
    sql = f"""
        SELECT c.chunk_id, 1 - (c.embedding <=> %(q)s) AS score, c.ticker, f.company,
               c.form, c.fiscal_year, c.fiscal_quarter, f.period_end::text, c.section_key,
               c.section_title, f.url, c.text
        FROM chunks c JOIN filings f USING (accession_number)
        WHERE {where}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
    """
    return sql, params


def vector_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    strategy: str,
    model: str,
    k: int = 5,
    filters: Filters | None = None,
) -> list[Hit]:
    sql, params = build_search_sql(filters or Filters())
    params |= {"q": HalfVector(query_vector), "strategy": strategy, "model": model, "k": k}
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        rows = conn.execute(sql, params).fetchall()
    return [Hit(*row) for row in rows]
