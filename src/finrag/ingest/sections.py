"""Group parsed Blocks into filing Items (sections).

Heading detection: a short text block starting with "Item <n>[A-C]". In a
10-Q, item numbers repeat across parts (Part I Item 1 = Financial Statements,
Part II Item 1 = Legal Proceedings), so 10-Q keys are prefixed with the part:
"I-2", "II-1A". 10-K item numbers are unique, so keys are just "1A", "7".

Table-of-contents dedupe: each heading usually appears twice, first in the
TOC and later in the body, so we keep the LAST occurrence of each key. Then,
to drop strays (a TOC entry for an item the body omits, a cross-reference that
happens to start a paragraph), we keep the longest run of headings that is in
canonical order (1 < 1A < 2 < ... ; Part I < Part II).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from finrag.ingest.models import Block, Section

ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\b\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)
PART_RE = re.compile(r"^part\s+(iv|i{1,3})\b", re.IGNORECASE)
MAX_HEADING_LEN = 200

TITLES_10K = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters "
    "and Issuer Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management "
    "and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
TITLES_10Q = {
    "I-1": "Financial Statements",
    "I-2": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "I-3": "Quantitative and Qualitative Disclosures About Market Risk",
    "I-4": "Controls and Procedures",
    "II-1": "Legal Proceedings",
    "II-1A": "Risk Factors",
    "II-2": "Unregistered Sales of Equity Securities and Use of Proceeds",
    "II-3": "Defaults Upon Senior Securities",
    "II-4": "Mine Safety Disclosures",
    "II-5": "Other Information",
    "II-6": "Exhibits",
}
PREAMBLE = "preamble"

# Some 10-Qs (e.g. JPMorgan) never print "Item 1." / "Item 2." in the body;
# they use a cross-reference index and start sections with plain titles, and
# put MD&A *before* the financial statements. Used only when a key is missing.
FALLBACK_TITLES_10Q = {
    "I-1": (
        "condensed consolidated statements of",
        "consolidated statements of income",
        "consolidated statements of operations",
        "consolidated balance sheets",
        "consolidated financial statements",
    ),
    "I-2": ("management's discussion and analysis", "executive overview"),
}
MAX_FALLBACK_LEN = 120

# Some 10-Ks (e.g. JPMorgan) make Items 7 and 8 one-paragraph pointers ("appears
# on pages 46-160") and append the whole annual report after Item 15. If a
# pointer Item is tiny and one section is huge, find the real headings inside
# the huge one and split them back out.
CROSS_REF_MAX_CHARS = 1_500
APPENDED_REPORT_MIN_CHARS = 100_000
RELOCATE_10K = {
    "7": ("management's discussion and analysis",),
    "8": ("report of independent registered public accounting firm",
          "consolidated statements of income", "consolidated statements of operations"),
}  # fmt: skip


@dataclass
class _Heading:
    index: int  # position in the block list
    key: str
    raw_title: str


def _find_headings(blocks: list[Block], form: str) -> list[_Heading]:
    is_10q = form.upper().startswith("10-Q")
    part = "I"
    headings = []
    for i, block in enumerate(blocks):
        if block.kind != "text" or len(block.text) > MAX_HEADING_LEN:
            continue
        if m := PART_RE.match(block.text):
            part = m.group(1).upper()
            # "PART II — OTHER INFORMATION Item 1. Legal Proceedings" in one block
            block_text = block.text[m.end() :].lstrip(" .:-–—")
            for word in ("FINANCIAL INFORMATION", "OTHER INFORMATION"):
                if block_text.upper().startswith(word):
                    block_text = block_text[len(word) :].lstrip(" .:-–—")
        else:
            block_text = block.text
        if m := ITEM_RE.match(block_text):
            item = m.group(1).upper()
            key = f"{part}-{item}" if is_10q else item
            headings.append(_Heading(i, key, m.group(2).strip()))
    return headings


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def _order_key(key: str) -> tuple[int, int, str]:
    """Canonical position: "II-1A" -> (2, 1, "A"); "7A" -> (0, 7, "A")."""
    part, _, item = key.rpartition("-")
    m = re.match(r"(\d+)([A-C]?)", item)
    return (_ROMAN.get(part, 0), int(m.group(1)), m.group(2))


def _longest_ordered_run(headings: list[_Heading]) -> list[_Heading]:
    """Longest subsequence (in document order) whose keys strictly increase."""
    n = len(headings)
    length, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if (
                _order_key(headings[j].key) < _order_key(headings[i].key)
                and length[j] + 1 > length[i]
            ):
                length[i], prev[i] = length[j] + 1, j
    if not n:
        return []
    i = max(range(n), key=length.__getitem__)
    run = []
    while i != -1:
        run.append(headings[i])
        i = prev[i]
    return run[::-1]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_part_marker(block: Block) -> bool:
    m = PART_RE.match(block.text)
    if block.kind != "text" or not m:
        return False
    rest = _norm(block.text[m.end() :])
    return rest in {"", "financialinformation", "otherinformation"}


def _body(blocks: list[Block], title: str) -> list[Block]:
    """Drop bare "PART II" lines, and the title line when the heading was split
    across two blocks ("Item 1A." / "Risk Factors")."""
    body = [b for b in blocks if not _is_part_marker(b)]
    if body and body[0].kind == "text" and _norm(body[0].text) == _norm(title):
        body = body[1:]
    return body


def _fallback_headings(blocks: list[Block], chosen: list[_Heading], form: str) -> list[_Heading]:
    """Find title-only headings for core 10-Q items that have no "Item" heading.
    Searches only before the first real heading, where Part I content lives."""
    if not form.upper().startswith("10-Q"):
        return []
    found = {h.key for h in chosen}
    limit = chosen[0].index if chosen else len(blocks)
    extra = []
    for key, aliases in FALLBACK_TITLES_10Q.items():
        if key in found:
            continue
        idx = _first_heading(blocks[:limit], aliases)
        if idx is not None:
            extra.append(_Heading(idx, key, blocks[idx].text))
    return extra


def _first_heading(blocks: list[Block], aliases: tuple[str, ...], start: int = 0) -> int | None:
    """Index of the first short text block starting with any alias (normalized)."""
    prefixes = [_norm(a) for a in aliases]
    for i in range(start, len(blocks)):
        b = blocks[i]
        short = b.kind == "text" and len(b.text) <= MAX_FALLBACK_LEN
        if short and any(_norm(b.text).startswith(p) for p in prefixes):
            return i
    return None


def _relocate_cross_referenced(sections: list[Section], form: str) -> list[Section]:
    """10-K only: move MD&A / financial statements out of an appended annual report."""
    if not form.upper().startswith("10-K") or not sections:
        return sections
    by_key = {s.key: s for s in sections}
    pointers = [
        k for k in RELOCATE_10K if k in by_key and len(by_key[k].text) <= CROSS_REF_MAX_CHARS
    ]
    big = max(sections, key=lambda s: len(s.text))
    if not pointers or big.key in RELOCATE_10K or len(big.text) < APPENDED_REPORT_MIN_CHARS:
        return sections

    starts = {}
    after = 0
    for key in ("7", "8"):
        if key in pointers:
            idx = _first_heading(big.blocks, RELOCATE_10K[key], after)
            if idx is not None:
                starts[key], after = idx, idx + 1
    if not starts:
        return sections

    cuts = sorted(starts.items(), key=lambda kv: kv[1])
    bounds = [i for _, i in cuts[1:]] + [len(big.blocks)]
    moved = []
    for (key, begin), end in zip(cuts, bounds):
        pointer = by_key[key]
        # Keep the pointer sentence; it states the page range, which is useful context.
        moved.append(Section(key, pointer.title, pointer.blocks + big.blocks[begin:end]))
    big.blocks = big.blocks[: cuts[0][1]]

    out = []
    for sec in sections:
        if sec.key in starts:
            continue  # replaced by the relocated version
        out.append(sec)
        if sec is big:
            out.extend(moved)
    return out


def split_sections(blocks: list[Block], form: str) -> list[Section]:
    """Split blocks into Sections. Blocks before the first heading (cover page,
    TOC) go into a "preamble" section."""
    last = {h.key: h for h in _find_headings(blocks, form)}  # later wins
    chosen = _longest_ordered_run(sorted(last.values(), key=lambda h: h.index))
    chosen = sorted(chosen + _fallback_headings(blocks, chosen, form), key=lambda h: h.index)

    titles = TITLES_10Q if form.upper().startswith("10-Q") else TITLES_10K
    sections: list[Section] = []
    first = chosen[0].index if chosen else len(blocks)
    if first > 0:
        sections.append(Section(PREAMBLE, "Cover Page", blocks[:first]))

    bounds = [h.index for h in chosen[1:]] + [len(blocks)]
    for heading, end in zip(chosen, bounds):
        title = titles.get(heading.key) or heading.raw_title.rstrip(".") or f"Item {heading.key}"
        sections.append(Section(heading.key, title, _body(blocks[heading.index + 1 : end], title)))
    return _relocate_cross_referenced(sections, form)
