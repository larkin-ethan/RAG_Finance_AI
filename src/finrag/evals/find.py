"""CLI: find evidence passages while hand-writing eval questions.

  python -m finrag.evals.find "membership fee" --ticker COST --form 10-K --year 2025

Prints matching chunks with the accession number and a snippet, so you can
copy an exact evidence phrase into evals/questions/human.jsonl. No database
or embeddings are needed; it searches the chunk files directly.
"""

from __future__ import annotations

import argparse
import json

from finrag.generate.citations import normalize
from finrag.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase")
    parser.add_argument("--ticker")
    parser.add_argument("--form")
    parser.add_argument("--year", type=int, help="fiscal year")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    needle = normalize(args.phrase)
    path = get_settings().data_dir / "chunks" / "section-512.jsonl"
    shown = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            if args.ticker and c["ticker"] != args.ticker.upper():
                continue
            if args.form and c["form"] != args.form.upper():
                continue
            if args.year and c["fiscal_year"] != args.year:
                continue
            text = normalize(c["text"])
            pos = text.find(needle)
            if pos < 0:
                continue
            q = f" Q{c['fiscal_quarter']}" if c["fiscal_quarter"] else ""
            print(f"\n{c['ticker']} {c['form']} FY{c['fiscal_year']}{q} · Item {c['section_key']} "
                  f"{c['section_title'][:50]}\n  accession: {c['accession_number']}")  # fmt: skip
            print(f"  ...{text[max(0, pos - 150) : pos + len(needle) + 200]}...")
            shown += 1
            if shown == args.limit:
                break
    if not shown:
        print("No matches. Try a shorter phrase, or drop a filter.")


if __name__ == "__main__":
    main()
