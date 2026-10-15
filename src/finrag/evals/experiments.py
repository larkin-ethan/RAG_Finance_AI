"""CLI: retrieval experiment sweep on the dev split (no LLM calls, free).

  python -m finrag.evals.experiments            # run all, print a comparison table
  python -m finrag.evals.experiments --only baseline hybrid

Prerequisites: each strategy below must be chunked and loaded, e.g.
  python -m finrag.chunking.run --max-tokens 256 --overlap 32
  python -m finrag.store.load --strategy section-256

Then run the full (LLM-graded) eval only for the baseline and the winner:
  python -m finrag.evals.run --name baseline --strategy fixed-512-64
  python -m finrag.evals.run --name final --strategy section-512 --method hybrid --filters auto --rerank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finrag.evals.dataset import load_questions
from finrag.evals.report import table
from finrag.evals.run import QUESTION_FILES, EvalConfig, run_eval

# name: (strategy, method, filters, rerank). One change at a time vs the row above
# where possible, so each row's delta is attributable to a single idea.
EXPERIMENTS = {
    "baseline": ("fixed-512-64", "vector", "none", False),  # naive RAG
    "section-aware": ("section-512", "vector", "none", False),
    "section-256": ("section-256", "vector", "none", False),
    "section-1024": ("section-1024", "vector", "none", False),
    "keyword-only": ("section-512", "keyword", "none", False),
    "hybrid": ("section-512", "hybrid", "none", False),
    "hybrid+filters": ("section-512", "hybrid", "auto", False),
    "hybrid+filters+rerank": ("section-512", "hybrid", "auto", True),
}


def main() -> None:
    from finrag.embed.embedder import Embedder
    from finrag.retrieve.retriever import RetrievalConfig, db_retriever
    from finrag.settings import get_settings, load_watchlist
    from finrag.store.db import connect

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    settings = get_settings()
    questions = load_questions(*QUESTION_FILES, split=args.split)
    conn, embedder = connect(settings.database_url), Embedder.from_settings(settings)
    tickers = load_watchlist(settings.watchlist_path).tickers

    summaries, names = [], []
    for name in args.only or EXPERIMENTS:
        strategy, method, filters, rerank = EXPERIMENTS[name]
        n_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE strategy = %s AND embed_model = %s",
            (strategy, embedder.client.model),
        ).fetchone()[0]
        if not n_chunks:
            print(f"skip {name}: strategy {strategy} isn't loaded")
            continue
        cfg = EvalConfig(name, strategy, mode="retrieval", method=method, filters=filters,
                         rerank=rerank)  # fmt: skip
        retrieve = db_retriever(
            conn, embedder, RetrievalConfig(strategy, method, filters, rerank), tickers
        )
        print(f"running {name} ({len(questions)} questions)...", flush=True)
        out = run_eval(questions, retrieve, cfg)
        path = Path("evals/results/sweep") / f"{name}-{args.split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        summaries.append(out["summary"])
        names.append(name)

    if summaries:
        print("\n" + table(summaries, names))


if __name__ == "__main__":
    main()
