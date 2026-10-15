"""CLI: try retrieval from the terminal.

  python -m finrag.retrieve.search "What was Apple's total net sales in fiscal 2024?" \
      --tickers AAPL --years 2024 --forms 10-K
"""

from __future__ import annotations

import argparse

from finrag.embed.embedder import Embedder
from finrag.retrieve.filters import Filters
from finrag.retrieve.vector import vector_search
from finrag.settings import get_settings
from finrag.store.db import connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--strategy", default="section-512")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--forms", nargs="*")
    args = parser.parse_args()

    settings = get_settings()
    embedder = Embedder.from_settings(settings)
    filters = Filters(tickers=args.tickers, fiscal_years=args.years, forms=args.forms)
    hits = vector_search(
        connect(settings.database_url),
        embedder.embed_query(args.question),
        args.strategy,
        embedder.client.model,
        args.k,
        filters,
    )
    for i, h in enumerate(hits, 1):
        period = f"FY{h.fiscal_year}" + (f" Q{h.fiscal_quarter}" if h.fiscal_quarter else "")
        print(
            f"\n[{i}] {h.score:.3f}  {h.ticker} {h.form} {period}  Item {h.section_key}: "
            f"{h.section_title[:60]}\n    {h.text[:400]!r}"
        )


if __name__ == "__main__":
    main()
