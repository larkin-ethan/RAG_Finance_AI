"""CLI: ask a question end to end (retrieve -> Gemini -> verify).

  python -m finrag.generate.ask "What was Apple's total net sales in fiscal 2024?" \
      --tickers AAPL --years 2024
"""

from __future__ import annotations

import argparse

from finrag.embed.embedder import Embedder
from finrag.generate.answer import answer_question
from finrag.generate.llm import make_llm
from finrag.retrieve.filters import Filters
from finrag.retrieve.vector import vector_search
from finrag.settings import get_settings
from finrag.store.db import connect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--strategy", default="section-512")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--forms", nargs="*")
    args = parser.parse_args()

    settings = get_settings()
    embedder = Embedder.from_settings(settings)
    conn = connect(settings.database_url)
    filters = Filters(tickers=args.tickers, fiscal_years=args.years, forms=args.forms)

    def retrieve(question: str):
        vector = embedder.embed_query(question)
        return vector_search(conn, vector, args.strategy, embedder.client.model, args.k, filters)

    ans = answer_question(args.question, retrieve, make_llm(settings))
    status = "grounded" if ans.grounded else "NOT fully grounded"
    print(
        f"\n{ans.text}\n\n[{status}]"
        + (f" abstained: {ans.abstain_reason}" if ans.abstain_reason else "")
    )
    for c in ans.citations:
        mark = "✓" if c.verified else "✗ quote not found"
        where = (
            f"{c.hit.ticker} {c.hit.form} FY{c.hit.fiscal_year} Item {c.hit.section_key}"
            if c.hit
            else "?"
        )
        print(f'  [{c.number}] {mark}  {where}\n      "{c.quote}"')
    if ans.unsupported_numbers:
        print(f"  Numbers not found verbatim in sources: {', '.join(ans.unsupported_numbers)}")
    print(
        f"\n{ans.model} | {ans.input_tokens:,} in / {ans.output_tokens:,} out tokens | "
        f"retrieval {ans.retrieval_s:.2f}s, generation {ans.generation_s:.2f}s"
        + (" (cached)" if ans.cached else "")
    )


if __name__ == "__main__":
    main()
