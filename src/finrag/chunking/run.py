"""CLI: chunk every parsed filing with both strategies.

Reads  data/parsed/{TICKER}/{accession}.json
Writes data/chunks/{strategy}.jsonl, one Chunk per line

Prints size stats and total tokens (= what embedding will cost), plus a
fiscal-period sanity check: each company should have a 10-K and Q1-Q3 per year.

Usage: python -m finrag.chunking.run [--max-tokens 512] [--overlap 64]
"""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict

from finrag.chunking.base import Chunk, Filing
from finrag.chunking.fixed import chunk_fixed
from finrag.chunking.section_aware import chunk_section_aware
from finrag.settings import get_settings


def report(name: str, chunks: list[Chunk]) -> None:
    sizes = [c.n_tokens for c in chunks]
    print(
        f"{name:18} {len(chunks):7,} chunks | tokens/chunk median {statistics.median(sizes):.0f}"
        f" p95 {sorted(sizes)[int(len(sizes) * 0.95)]} max {max(sizes)}"
        f" | total {sum(sizes) / 1e6:.1f}M tokens"
    )


def check_fiscal_periods(filings: list[Filing]) -> None:
    periods = defaultdict(list)
    for f in filings:
        periods[(f.ticker, f.fiscal_year)].append(f.fiscal_quarter or 4)
    problems = [(k, sorted(v)) for k, v in periods.items() if len(set(v)) != len(v)]
    for (ticker, fy), quarters in sorted(problems):
        print(f"  !! {ticker} FY{fy} has duplicate periods: {quarters}")
    print(f"Fiscal check: {len(periods)} company-years, {len(problems)} with duplicates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    files = sorted((settings.data_dir / "parsed").glob("*/*.json"))
    filings = [Filing.load(p) for p in files]
    check_fiscal_periods(filings)

    out_dir = settings.data_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = {
        f"fixed-{args.max_tokens}-{args.overlap}": lambda f: chunk_fixed(
            f, args.max_tokens, args.overlap
        ),
        f"section-{args.max_tokens}": lambda f: chunk_section_aware(f, args.max_tokens),
    }
    for name, chunker in strategies.items():
        chunks = [c for f in filings for c in chunker(f)]
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(asdict(c)) + "\n")
        report(name, chunks)


if __name__ == "__main__":
    main()
