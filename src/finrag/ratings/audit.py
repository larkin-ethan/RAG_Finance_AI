"""CLI: XBRL tag coverage per company, i.e. which concept fed each metric and
what's missing. Tag usage varies a lot between filers; a silently missing
input (e.g. debt) is how a scorecard ends up confidently wrong.

  python -m finrag.ratings.audit [--as-of 2025-03-01] [--show-concepts]
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.xbrl import load_company_facts
from finrag.ratings.fundamentals import RATING_METRICS, FactStore
from finrag.settings import get_settings, load_watchlist

# Metrics the goal recipes need (banks are allowed to miss the industrial ones).
REQUIRED = ["revenue", "net_income", "eps_diluted", "operating_cash_flow", "capex", "total_assets",
            "equity", "cash", "total_debt", "dividends_paid", "diluted_shares", "ebit"]  # fmt: skip
NOT_FOR_BANKS = {"capex", "total_debt", "ebit"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--show-concepts", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    wl = load_watchlist(settings.watchlist_path)
    client = EdgarClient(settings.sec_user_agent)
    for ticker in wl.tickers:
        ciks = [client.get_cik(ticker), *(f"{int(c):010d}" for c in wl.extra_ciks.get(ticker, []))]
        store = FactStore(
            *(load_company_facts(client, c, settings.data_dir / "xbrl") for c in ciks)
        )
        snap = store.snapshot(args.as_of)
        have = set(snap.ttm) | set(snap.latest)
        missing = [
            m for m in REQUIRED if m not in have and not (snap.is_financial and m in NOT_FOR_BANKS)
        ]
        kind = "bank" if snap.is_financial else "company"
        print(f"{ticker:6} {kind:7} period {snap.period_end}  missing: {', '.join(missing) or '-'}")
        if args.show_concepts:
            for metric, concept in sorted(store.concepts_used(args.as_of).items()):
                if metric in RATING_METRICS or metric == "shares_outstanding":
                    print(f"         {metric:22} <- {concept}")
    print(json.dumps({"as_of": str(args.as_of)}))


if __name__ == "__main__":
    main()
