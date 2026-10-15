"""Shared chunking types: the Chunk record, filing loader, token counting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from finrag.ingest.fiscal import fiscal_period
from finrag.ingest.models import Block, Section

# Approximate tokenizer: words, numbers and individual punctuation marks.
# "$1,234.5" -> ["$", "1", ",", "234", ".", "5"]. That's close to how BPE
# tokenizers treat financial text. Calibrated against real Voyage counts on Day 4.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return sum(1 for _ in TOKEN_RE.finditer(text))


@dataclass
class Filing:
    """A parsed filing plus the metadata every chunk inherits."""

    ticker: str
    company: str
    form: str
    accession_number: str
    filing_date: date
    period_end: date
    fiscal_year: int
    fiscal_quarter: int | None
    url: str
    sections: list[Section]
    fiscal_year_end: str = "1231"  # nominal "MMDD" from EDGAR

    @property
    def period_label(self) -> str:
        q = f" Q{self.fiscal_quarter}" if self.fiscal_quarter else ""
        return f"FY{self.fiscal_year}{q}"

    @classmethod
    def load(cls, path: Path) -> Filing:
        d = json.loads(path.read_text(encoding="utf-8"))
        filing_date = date.fromisoformat(d["filing_date"])
        # reportDate is occasionally blank; filings land ~4-8 weeks after period end.
        period_end = (
            date.fromisoformat(d["report_date"])
            if d.get("report_date")
            else filing_date - timedelta(days=45)
        )
        if "fiscal_year_end" not in d:
            # Silently assuming December mislabels every non-calendar filer.
            raise ValueError(f"{path} has no fiscal_year_end; re-run `make fetch parse`")
        fy, fq = fiscal_period(period_end, d["fiscal_year_end"], d["form"])
        sections = [
            Section(s["key"], s["title"], [Block(b["kind"], b["text"]) for b in s["blocks"]])
            for s in d["sections"]
        ]
        return cls(
            ticker=d["ticker"],
            company=d.get("company", d["ticker"]),
            form=d["form"],
            accession_number=d["accession_number"],
            filing_date=filing_date,
            period_end=period_end,
            fiscal_year=fy,
            fiscal_quarter=fq,
            url=d.get("url", ""),
            sections=sections,
            fiscal_year_end=d["fiscal_year_end"],
        )


@dataclass
class Chunk:
    chunk_id: str  # "{accession}:{strategy}:{n:04d}"
    strategy: str
    ticker: str
    company: str
    form: str
    accession_number: str
    filing_date: str
    period_end: str
    fiscal_year: int
    fiscal_quarter: int | None
    section_key: str
    section_title: str
    url: str  # EDGAR link to the source document
    text: str  # filing text, shown to the LLM and cited
    embed_text: str  # what gets embedded (may add a context header)
    n_tokens: int  # of embed_text

    @classmethod
    def build(
        cls, filing: Filing, strategy: str, n: int, section: Section, text: str, embed_text: str
    ) -> Chunk:
        return cls(
            chunk_id=f"{filing.accession_number}:{strategy}:{n:04d}",
            strategy=strategy,
            ticker=filing.ticker,
            company=filing.company,
            form=filing.form,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date.isoformat(),
            period_end=filing.period_end.isoformat(),
            fiscal_year=filing.fiscal_year,
            fiscal_quarter=filing.fiscal_quarter,
            section_key=section.key,
            section_title=section.title,
            url=filing.url,
            text=text,
            embed_text=embed_text,
            n_tokens=count_tokens(embed_text),
        )
