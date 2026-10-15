"""SEC XBRL "company facts": the machine-readable financials companies file
with each 10-K/10-Q. Used as an independent source of truth for eval answers,
so our gold numbers don't come from our own parser.

  https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  facts["us-gaap"][concept]["units"]["USD"] = [
      {"start": "2023-10-01", "end": "2024-09-28", "val": 391035000000,
       "accn": "0000320193-24-000123", "form": "10-K", "fp": "FY", ...}, ...]

A 10-K reports several years of each line item under one accession, so a fact
belongs to a filing only when the fact's end date equals that filing's period end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from finrag.ingest.edgar_client import EdgarClient

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


@dataclass(frozen=True)
class Metric:
    key: str
    label: str  # used in question text
    concepts: tuple[str, ...]  # XBRL tags to try, in order (companies differ)
    unit: str  # "USD" or "USD/shares"
    instant: bool = False  # balance-sheet item (no start date)


METRICS = [
    Metric("revenue", "total revenue",
           ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
            "RevenuesNetOfInterestExpense", "SalesRevenueNet"), "USD"),
    Metric("net_income", "net income", ("NetIncomeLoss",), "USD"),
    Metric("eps_diluted", "diluted earnings per share", ("EarningsPerShareDiluted",), "USD/shares"),
    Metric("operating_income", "operating income", ("OperatingIncomeLoss",), "USD"),
    Metric("total_assets", "total assets", ("Assets",), "USD", instant=True),
    Metric("cash", "cash and cash equivalents",
           ("CashAndCashEquivalentsAtCarryingValue",), "USD", instant=True),
]  # fmt: skip


@dataclass(frozen=True)
class Fact:
    ticker: str
    metric: Metric
    concept: str
    value: float
    start: date | None
    end: date
    accession_number: str
    form: str

    @property
    def days(self) -> int | None:
        return (self.end - self.start).days if self.start else None


def load_company_facts(client: EdgarClient, cik: str, cache_dir: Path) -> dict:
    """Fetch (or read cached) companyfacts JSON for one CIK."""
    path = cache_dir / f"CIK{cik}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(client._get(FACTS_URL.format(cik=cik)).text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_facts(ticker: str, company_facts: dict) -> list[Fact]:
    """All 10-K/10-Q facts for METRICS, using the first concept a company reports."""
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    out = []
    for metric in METRICS:
        concept = next(
            (c for c in metric.concepts if metric.unit in gaap.get(c, {}).get("units", {})), None
        )
        if concept is None:
            continue
        for f in gaap[concept]["units"][metric.unit]:
            if f.get("form") not in ("10-K", "10-Q"):
                continue
            out.append(
                Fact(
                    ticker=ticker,
                    metric=metric,
                    concept=concept,
                    value=float(f["val"]),
                    start=date.fromisoformat(f["start"]) if f.get("start") else None,
                    end=date.fromisoformat(f["end"]),
                    accession_number=f["accn"],
                    form=f["form"],
                )
            )
    return out
