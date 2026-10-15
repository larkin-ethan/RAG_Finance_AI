"""Extract metadata filters from the question text (rule-based, no LLM).

Free, deterministic, and testable, with no Gemini quota spent per query. It
reads ONLY the question, never eval gold labels.

  "How did Apple's net income change from fiscal 2023 to fiscal 2024?"
    -> tickers [AAPL], fiscal_years [2023, 2024]

Year caution: a fiscal year filter is applied only for explicit fiscal
phrasing ("fiscal 2024", "FY2024", "fiscal year 2024"). "The quarter ended
July 27, 2025" is calendar-dated, and for NVIDIA that's fiscal 2026, so a
calendar year must never become a fiscal_year filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from finrag.retrieve.filters import Filters

FISCAL_YEAR = re.compile(r"\b(?:fiscal(?:\s+year)?|FY)\s*'?(\d{4}|\d{2})\b", re.IGNORECASE)
FORM = re.compile(r"\b10-?([KQ])\b", re.IGNORECASE)
ANNUAL_HINTS = re.compile(r"\bannual report\b", re.IGNORECASE)

DEFAULT_ALIASES = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "NVDA": ["NVIDIA", "Nvidia"],
    "JPM": ["JPMorgan", "JP Morgan", "J.P. Morgan", "JPMorgan Chase"],
    "JNJ": ["Johnson & Johnson", "Johnson and Johnson", "J&J"],
    "XOM": ["ExxonMobil", "Exxon Mobil", "Exxon"],
    "KO": ["Coca-Cola", "Coca Cola", "Coke"],
    "COST": ["Costco"],
}


@dataclass
class QueryParser:
    aliases: dict[str, list[str]]  # ticker -> names it may appear as

    @classmethod
    def for_tickers(cls, tickers: list[str], extra: dict[str, list[str]] | None = None):
        aliases = {t: DEFAULT_ALIASES.get(t, []) + (extra or {}).get(t, []) for t in tickers}
        return cls(aliases)

    def tickers(self, question: str) -> list[str]:
        found = []
        for ticker, names in self.aliases.items():
            # Ticker symbols must be uppercase words ("COST" yes, "cost" no).
            if re.search(rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])", question):
                found.append(ticker)
                continue
            for name in names:
                if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.IGNORECASE):
                    found.append(ticker)
                    break
        return found

    @staticmethod
    def fiscal_years(question: str) -> list[int]:
        years = []
        for raw in FISCAL_YEAR.findall(question):
            year = int(raw) + (2000 if len(raw) == 2 else 0)
            if year not in years:
                years.append(year)
        return years

    @staticmethod
    def forms(question: str) -> list[str]:
        forms = {f"10-{m.upper()}" for m in FORM.findall(question)}
        if ANNUAL_HINTS.search(question):
            forms.add("10-K")
        return sorted(forms)

    def parse(self, question: str) -> Filters:
        return Filters(
            tickers=self.tickers(question) or None,
            fiscal_years=self.fiscal_years(question) or None,
            forms=self.forms(question) or None,
        )
