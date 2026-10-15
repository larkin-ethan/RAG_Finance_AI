"""Verify that an answer is grounded in its cited sources.

Two checks, both deterministic (no LLM):
  1. Quote check: every quote must appear in the chunk it cites. Whitespace,
     quote/dash styles, case, and markdown table pipes are normalized, and
     "..." in a quote may skip text (each piece must appear, in order).
  2. Number check: every number in the answer must match a number in a cited
     chunk, after unit scaling. "$391.0 billion" matches "391,035" from a table
     "in millions", within rounding tolerance.

Numbers that fail check 2 are reported, not silently dropped. They may be
computed (e.g. a growth rate the model derived), but the UI should say
they weren't found verbatim.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "|": " "})
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")

SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "mm": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "trillions": 1e12, "tn": 1e12,
}  # fmt: skip
NUMBER_RE = re.compile(
    r"(?P<neg>\()?(?P<cur>\$)?(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?\)?"
    r"(?P<pct>\s?%)?(?:\s?(?P<scale>thousand|millions?|billions?|trillions?|bn|mm|tn|[kmb])\b)?",
    re.IGNORECASE,
)
# Labels that contain digits but aren't quantities: [1], FY2024, Q3, Item 7A, Note 14.
_NOT_QUANTITIES = re.compile(
    r"\[\d+\]|\bFY\s?\d{2,4}\b|\bQ[1-4]\b|\bItem\s+\d+[A-C]?\b|\bNote\s+\d+\b|\b10-[KQ]\b",
    re.IGNORECASE,
)
# Dates are context, not claims to verify: "September 28, 2024", "2024-09-28".
_DATES = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_TRANSLATE)).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    haystack = normalize(text)
    position = 0
    pieces = [p for p in _ELLIPSIS.split(normalize(quote)) if p.strip(" .")]
    if not pieces:
        return False
    for piece in pieces:
        found = haystack.find(piece.strip(), position)
        if found < 0:
            return False
        position = found + len(piece.strip())
    return True


@dataclass(frozen=True)
class Quantity:
    raw: str
    value: float  # absolute value, scaled
    tolerance: float  # half a unit in the last stated digit, scaled
    is_percent: bool


def extract_numbers(text: str) -> list[Quantity]:
    text = _DATES.sub(" ", _NOT_QUANTITIES.sub(" ", text))
    out = []
    for m in NUMBER_RE.finditer(text):
        digits = m["num"].replace(",", "")
        decimals = m["dec"] or ""
        value = float(digits + decimals)
        has_marker = bool(m["cur"] or m["pct"] or m["scale"] or decimals or "," in m["num"])
        # Bare small integers ("3 segments") and bare years aren't verifiable claims.
        if not has_marker and (value <= 12 or (1900 <= value <= 2100 and len(digits) == 4)):
            continue
        scale = SCALES.get((m["scale"] or "").lower(), 1.0)
        unit = 10 ** -(len(decimals) - 1) if decimals else 1.0
        out.append(
            Quantity(m.group(0).strip(), abs(value) * scale, unit * scale / 2, bool(m["pct"]))
        )
    return out


def _matches(answer_q: Quantity, source_q: Quantity) -> bool:
    if answer_q.is_percent != source_q.is_percent:
        return False
    scales = (1.0,) if answer_q.is_percent else (1.0, 1e3, 1e6, 1e9)
    for s in scales:  # tables omit units ("in millions"), so try each scale
        candidate = source_q.value * s
        # Precision the answer states, plus float noise. A 0.5% slack would let
        # EPS $6.11 pass as $6.08.
        tol = max(answer_q.tolerance, 0.001 * candidate, 1e-9)
        if math.isclose(answer_q.value, candidate, abs_tol=tol):
            return True
    return False


def unsupported_numbers(answer: str, sources: list[str]) -> list[str]:
    available = [q for text in sources for q in extract_numbers(text)]
    return [q.raw for q in extract_numbers(answer) if not any(_matches(q, s) for s in available)]


@dataclass
class CitationCheck:
    source: int  # 1-based index into the retrieved hits
    quote: str
    valid_source: bool
    quote_found: bool


@dataclass
class Verification:
    citations: list[CitationCheck] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    uncited_markers: list[int] = field(default_factory=list)  # [n] in text with no quote

    @property
    def verified_sources(self) -> set[int]:
        return {c.source for c in self.citations if c.valid_source and c.quote_found}

    @property
    def grounded(self) -> bool:
        return (
            bool(self.citations)
            and all(c.valid_source and c.quote_found for c in self.citations)
            and not self.unsupported_numbers
        )


def verify(answer: str, citations: list[dict], source_texts: list[str]) -> Verification:
    result = Verification()
    for c in citations:
        idx, quote = int(c.get("source", 0)), str(c.get("quote", ""))
        valid = 1 <= idx <= len(source_texts)
        found = valid and quote_in_text(quote, source_texts[idx - 1])
        result.citations.append(CitationCheck(idx, quote, valid, found))

    cited = sorted({c.source for c in result.citations if c.valid_source})
    result.unsupported_numbers = unsupported_numbers(answer, [source_texts[i - 1] for i in cited])
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    result.uncited_markers = sorted(markers - {c.source for c in result.citations})
    return result
