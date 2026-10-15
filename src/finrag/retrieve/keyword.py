"""Keyword retrieval with Postgres full-text search (the `tsv` column).

Natural questions are long, and plainto_tsquery ANDs every term, so almost
nothing matches. We OR the terms instead ('apple' | 'revenu' | '2024') and
let ts_rank_cd reward chunks that match more of them, close together. This is
Postgres FTS ranking, not true BM25 (no term-frequency saturation), so results
call it "keyword (Postgres FTS)".
"""

from __future__ import annotations

import psycopg

from finrag.retrieve.filters import Filters
from finrag.retrieve.vector import Hit

OR_QUERY = "NULLIF(replace(plainto_tsquery('english', %(q)s)::text, '&', '|'), '')::tsquery"


def build_keyword_sql(filters: Filters) -> tuple[str, dict]:
    clauses, params = filters.where()
    where = " AND ".join(
        ["c.strategy = %(strategy)s", "c.embed_model = %(model)s", f"c.tsv @@ {OR_QUERY}", *clauses]
    )
    sql = f"""
        SELECT c.chunk_id, ts_rank_cd(c.tsv, {OR_QUERY}) AS score, c.ticker, f.company,
               c.form, c.fiscal_year, c.fiscal_quarter, f.period_end::text, c.section_key,
               c.section_title, f.url, c.text
        FROM chunks c JOIN filings f USING (accession_number)
        WHERE {where}
        ORDER BY score DESC
        LIMIT %(k)s
    """
    return sql, params


def keyword_search(
    conn: psycopg.Connection,
    question: str,
    strategy: str,
    model: str,
    k: int = 30,
    filters: Filters | None = None,
) -> list[Hit]:
    sql, params = build_keyword_sql(filters or Filters())
    params |= {"q": question, "strategy": strategy, "model": model, "k": k}
    return [Hit(*row) for row in conn.execute(sql, params).fetchall()]
