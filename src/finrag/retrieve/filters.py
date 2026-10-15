"""Metadata filters for retrieval (company, fiscal year/quarter, form, section)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Filters:
    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    fiscal_quarters: list[int] | None = None  # use 0 for 10-K (annual) periods
    forms: list[str] | None = None
    section_keys: list[str] | None = None

    def where(self, alias: str = "c") -> tuple[list[str], dict]:
        """SQL WHERE fragments plus their parameters. Values always go through
        parameters; only these fixed column names are ever inserted as text.
        Columns are qualified with the chunks table's alias, because joined
        tables (e.g. filings) share names like ticker and fiscal_year."""
        a = f"{alias}." if alias else ""
        clauses, params = [], {}
        if self.tickers:
            clauses.append(f"{a}ticker = ANY(%(tickers)s)")
            params["tickers"] = [t.upper() for t in self.tickers]
        if self.fiscal_years:
            clauses.append(f"{a}fiscal_year = ANY(%(fiscal_years)s)")
            params["fiscal_years"] = self.fiscal_years
        if self.fiscal_quarters:
            clauses.append(f"COALESCE({a}fiscal_quarter, 0) = ANY(%(fiscal_quarters)s)")
            params["fiscal_quarters"] = self.fiscal_quarters
        if self.forms:
            clauses.append(f"{a}form = ANY(%(forms)s)")
            params["forms"] = [f.upper() for f in self.forms]
        if self.section_keys:
            clauses.append(f"{a}section_key = ANY(%(section_keys)s)")
            params["section_keys"] = self.section_keys
        return clauses, params
