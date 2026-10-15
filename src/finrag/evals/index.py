"""Export/import the retrieval index (filings + one strategy's chunks with
embeddings) as a single gzipped file, so CI can run the retrieval eval without
re-embedding the corpus.

  python -m finrag.evals.index export --strategy section-512 --out finrag-index.tsv.gz
  python -m finrag.evals.index import finrag-index.tsv.gz

Uses Postgres COPY through psycopg, so it needs no pg_dump/psql install
(works the same on Windows). Attach the file to a GitHub release tagged
`eval-index` and CI downloads it.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from finrag.settings import get_settings
from finrag.store.db import connect

CHUNK_COLS = (
    "chunk_id, embed_model, strategy, accession_number, ticker, form, fiscal_year, "
    "fiscal_quarter, section_key, section_title, text, embed_text, n_tokens, embedding"
)
MARK = b"--- FINRAG CHUNKS ---\n"


def export(strategy: str, model: str, out: Path) -> None:
    conn = connect(get_settings().database_url)
    with gzip.open(out, "wb") as fh:
        with conn.cursor().copy(
            "COPY (SELECT f.* FROM filings f WHERE EXISTS (SELECT 1 FROM chunks c "
            "WHERE c.accession_number = f.accession_number AND c.strategy = %s "
            "AND c.embed_model = %s)) TO STDOUT", (strategy, model),
        ) as copy:  # fmt: skip
            for block in copy:
                fh.write(bytes(block))
        fh.write(MARK)
        with conn.cursor().copy(
            f"COPY (SELECT {CHUNK_COLS} FROM chunks WHERE strategy = %s AND embed_model = %s) "
            "TO STDOUT", (strategy, model),
        ) as copy:  # fmt: skip
            for block in copy:
                fh.write(bytes(block))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def import_(path: Path) -> None:
    conn = connect(get_settings().database_url, init=True)
    data = gzip.decompress(path.read_bytes())
    filings, chunks = data.split(MARK, 1)
    with conn.transaction():
        with conn.cursor().copy("COPY filings FROM STDIN") as copy:
            copy.write(filings)
        with conn.cursor().copy(f"COPY chunks ({CHUNK_COLS}) FROM STDIN") as copy:
            copy.write(chunks)
    n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Imported {path.name}: {n:,} chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--strategy", default="section-512")
    e.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    e.add_argument("--out", type=Path, default=Path("finrag-index.tsv.gz"))
    i = sub.add_parser("import")
    i.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.cmd == "export":
        export(args.strategy, args.model, args.out)
    else:
        import_(args.path)


if __name__ == "__main__":
    main()
