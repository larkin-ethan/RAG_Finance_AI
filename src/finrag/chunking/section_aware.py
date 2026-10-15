"""Section-aware chunking: respect the filing's structure.

* Chunks never cross an Item boundary, so every chunk has one section label.
* Whole blocks (paragraphs, tables) are packed together up to max_tokens.
* An oversized table is split between rows, and each piece repeats the
  table's header row, so "2024 | 2023" stays attached to the numbers.
* An oversized paragraph is split between sentences.
* A short lead-in like "Net sales by category (in millions):" moves with
  the table that follows it, since it carries the units.
* The embedded text gets a context header (company, form, fiscal period,
  section). Chunk text alone often never says which company or year it's
  about ("Revenue increased 6%..."), so the header lets retrieval match
  "Apple 2024 revenue" to the right chunk.
"""

from __future__ import annotations

import re

from finrag.chunking.base import TOKEN_RE, Chunk, Filing, count_tokens

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“$])")
LEAD_IN_MAX_TOKENS = 60


def context_header(filing: Filing, section_title: str) -> str:
    return (
        f"{filing.company} ({filing.ticker}) | Form {filing.form} | {filing.period_label} "
        f"(period ended {filing.period_end.isoformat()}) | {section_title}"
    )


def _hard_split(text: str, max_tokens: int) -> list[str]:
    spans = [m.span() for m in TOKEN_RE.finditer(text)]
    return [
        text[spans[i][0] : spans[min(i + max_tokens, len(spans)) - 1][1]]
        for i in range(0, len(spans), max_tokens)
    ]


def _pack(units: list[str], max_tokens: int, sep: str, prefix: str = "") -> list[str]:
    """Greedily join units (each <= budget) into pieces of <= max_tokens."""
    budget = max_tokens - count_tokens(prefix)
    pieces, current, size = [], [], 0
    for unit in units:
        n = count_tokens(unit)
        if current and size + n > budget:
            pieces.append(prefix + sep.join(current))
            current, size = [], 0
        current.append(unit)
        size += n
    if current:
        pieces.append(prefix + sep.join(current))
    return pieces


def split_text(text: str, max_tokens: int) -> list[str]:
    sentences = []
    for sentence in SENTENCE_END.split(text):
        sentences += (
            _hard_split(sentence, max_tokens) if count_tokens(sentence) > max_tokens else [sentence]
        )
    return _pack(sentences, max_tokens, " ")


def split_table(markdown: str, max_tokens: int) -> list[str]:
    lines = markdown.split("\n")
    header = "\n".join(lines[:2]) + "\n"  # column-header row + |---| separator
    if count_tokens(header) > max_tokens // 2:  # pathological header: plain split
        return split_text(markdown, max_tokens)
    budget = max_tokens - count_tokens(header)
    rows = []
    for row in lines[2:]:
        rows += _hard_split(row, budget) if count_tokens(row) > budget else [row]
    return _pack(rows, max_tokens, "\n", prefix=header)


def chunk_section_aware(filing: Filing, max_tokens: int = 512) -> list[Chunk]:
    strategy = f"section-{max_tokens}"
    chunks: list[Chunk] = []

    for section in filing.sections:
        header = context_header(filing, section.title)
        budget = max_tokens - count_tokens(header)

        # 1. Break the section into (kind, text) pieces that each fit the budget.
        pieces: list[tuple[str, str]] = []
        for block in section.blocks:
            if count_tokens(block.text) <= budget:
                pieces.append((block.kind, block.text))
            elif block.kind == "table":
                pieces += [("table", t) for t in split_table(block.text, budget)]
            else:
                pieces += [("text", t) for t in split_text(block.text, budget)]

        # 2. Pack pieces into chunks without crossing the budget.
        groups: list[list[str]] = []
        current: list[tuple[str, str]] = []
        size = 0
        for kind, text in pieces:
            n = count_tokens(text)
            if current and size + n > budget:
                carry = []
                last_kind, last_text = current[-1]
                lead_in = count_tokens(last_text)
                is_lead_in = last_kind == "text" and lead_in <= LEAD_IN_MAX_TOKENS
                fits = lead_in + n <= budget
                if kind == "table" and is_lead_in and fits and len(current) > 1:
                    carry = [current.pop()]  # keep "(in millions):" with its table
                groups.append([t for _, t in current])
                current = carry
                size = sum(count_tokens(t) for _, t in current)
            current.append((kind, text))
            size += n
        if current:
            groups.append([t for _, t in current])

        for group in groups:
            text = "\n\n".join(group)
            # Tiny sections are kept on purpose: "Mine Safety Disclosures: Not
            # applicable." is a real, citable answer once the header is attached.
            if not text.strip():
                continue
            chunk = Chunk.build(filing, strategy, len(chunks), section, text, f"{header}\n\n{text}")
            chunks.append(chunk)
    return chunks
