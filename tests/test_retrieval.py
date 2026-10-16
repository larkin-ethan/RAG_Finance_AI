import os

import psycopg
import pytest

from finrag.retrieve.filters import Filters
from finrag.retrieve.vector import build_search_sql


def test_no_filters_only_strategy_and_model():
    sql, params = build_search_sql(Filters())
    assert "c.strategy = %(strategy)s AND c.embed_model = %(model)s" in sql
    assert params == {}


def test_filters_are_parameterized_and_normalized():
    sql, params = build_search_sql(
        Filters(tickers=["aapl"], fiscal_years=[2024], forms=["10-k"], fiscal_quarters=[0])
    )
    assert "c.ticker = ANY(%(tickers)s)" in sql
    assert "COALESCE(c.fiscal_quarter, 0) = ANY(%(fiscal_quarters)s)" in sql
    assert params == {
        "tickers": ["AAPL"],
        "fiscal_years": [2024],
        "forms": ["10-K"],
        "fiscal_quarters": [0],
    }
    assert "aapl" not in sql.lower().replace("%(tickers)s", "")  # values never inlined


# ---------- integration: needs `make db` (skipped otherwise) ----------

DB_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://finrag:finrag@localhost:5432/finrag")


def _db_available() -> bool:
    try:
        psycopg.connect(DB_URL, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


@pytest.mark.skipif(not _db_available(), reason="Postgres not running (make db)")
def test_filtered_search_returns_k_rows_from_skewed_table():
    """Iterative scan: AAPL is 2% of rows, yet we still get k=5 AAPL hits."""
    from pgvector import HalfVector

    from finrag.retrieve.vector import vector_search
    from finrag.store.db import connect

    conn = connect(DB_URL, init=True)
    acc = "test-0000"
    with conn.transaction():
        conn.execute("DELETE FROM filings WHERE accession_number LIKE 'test-%'")
        for i, ticker in enumerate(["AAPL"] * 10 + ["MSFT"] * 490):
            a = f"{acc}-{ticker}"
            conn.execute(
                "INSERT INTO filings VALUES (%s,%s,%s,'10-K','2024-11-01','2024-09-28',2024,NULL,'u')"
                " ON CONFLICT DO NOTHING",
                (a, ticker, ticker),
            )
            vec = [1.0 if j == i % 1024 else 0.0 for j in range(1024)]
            conn.execute(
                "INSERT INTO chunks (chunk_id, embed_model, strategy, accession_number, ticker,"
                " form, fiscal_year, section_key, section_title, text, embed_text, n_tokens,"
                " embedding) VALUES (%s,'test-model','test',%s,%s,'10-K',2024,'7','MD&A',"
                " 't','t',1,%s)",
                (f"test-{i}", a, ticker, HalfVector(vec)),
            )
    try:
        query = [1.0 if j == 300 else 0.0 for j in range(1024)]  # nearest to an MSFT row
        hits = vector_search(conn, query, "test", "test-model", 5, Filters(tickers=["AAPL"]))
        assert len(hits) == 5 and {h.ticker for h in hits} == {"AAPL"}
    finally:
        conn.execute("DELETE FROM filings WHERE accession_number LIKE 'test-%'")


@pytest.mark.skipif(not _db_available(), reason="Postgres not running (make db)")
def test_eval_index_export_import_roundtrip(tmp_path, monkeypatch):
    from pgvector import HalfVector

    from finrag.evals import index
    from finrag.settings import get_settings
    from finrag.store.db import connect

    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("SEC_USER_AGENT", "t t@example.com")
    get_settings.cache_clear()
    conn = connect(DB_URL, init=True)
    acc = "test-idx-0001"

    def wipe():
        conn.execute("DELETE FROM filings WHERE accession_number LIKE 'test-idx-%'")

    wipe()
    conn.execute(
        "INSERT INTO filings VALUES (%s,'IDX','Index Co','10-K','2024-11-01','2024-09-28',2024,NULL,'u')",
        (acc,),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, embed_model, strategy, accession_number, ticker, form,"
        " fiscal_year, section_key, section_title, text, embed_text, n_tokens, embedding)"
        " VALUES (%s,'idx-model','idx-strat',%s,'IDX','10-K',2024,'7','MD&A',"
        " %s,'e',3,%s)",
        (
            f"{acc}:idx-strat:0000",
            acc,
            'Tab\there, newline\nhere, quote "x"',
            HalfVector([0.5] * 1024),
        ),
    )
    try:
        out = tmp_path / "idx.tsv.gz"
        index.export("idx-strat", "idx-model", out)
        wipe()
        index.import_(out)
        text = conn.execute(
            "SELECT text FROM chunks WHERE chunk_id = %s", (f"{acc}:idx-strat:0000",)
        ).fetchone()[0]
        assert text == 'Tab\there, newline\nhere, quote "x"'  # escaping survived
    finally:
        wipe()
        get_settings.cache_clear()


@pytest.mark.skipif(not _db_available(), reason="Postgres not running (make db)")
def test_positions_are_stored_encrypted_and_audited():
    from datetime import date

    from cryptography.fernet import Fernet

    from finrag.ratings.positions import Lot, PositionsRepo
    from finrag.security.crypto import FieldCipher
    from finrag.store.db import connect

    conn = connect(DB_URL, init=True)
    repo = PositionsRepo(conn, FieldCipher(Fernet.generate_key().decode()))
    pid = repo.add(Lot("ZZTEST", 7, 123.45, date(2022, 2, 2), "Roth"))
    try:
        raw = conn.execute("SELECT lot_enc FROM positions WHERE id = %s", (pid,)).fetchone()[0]
        assert "ZZTEST" not in raw and "123.45" not in raw and "Roth" not in raw
        assert next(lot for lot in repo.list() if lot.id == pid).ticker == "ZZTEST"
        repo.update(pid, Lot("ZZTEST", 8, 123.45, date(2022, 2, 2)))
        assert next(lot for lot in repo.list() if lot.id == pid).shares == 8
        actions = [r[0] for r in conn.execute(
            "SELECT action FROM audit_log WHERE position_id = %s ORDER BY id", (pid,)).fetchall()]  # fmt: skip
        assert actions == ["add", "update"]
    finally:
        repo.delete(pid)
        conn.execute("DELETE FROM audit_log WHERE position_id = %s", (pid,))
