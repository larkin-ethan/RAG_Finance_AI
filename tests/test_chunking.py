from datetime import date

import pytest

from finrag.chunking.base import Filing, count_tokens
from finrag.chunking.fixed import chunk_fixed
from finrag.chunking.section_aware import chunk_section_aware, split_table
from finrag.ingest.models import Block, Section


def make_filing(sections: list[Section], form: str = "10-K") -> Filing:
    return Filing(
        ticker="ACME",
        company="ACME Corp",
        form=form,
        accession_number="0000000000-24-000001",
        filing_date=date(2024, 11, 1),
        period_end=date(2024, 9, 28),
        fiscal_year=2024,
        fiscal_quarter=None,
        url="",
        sections=sections,
    )


def words(n: int, word: str = "alpha") -> str:
    return " ".join([word] * n)


def table(n_rows: int) -> str:
    rows = "\n".join(f"| Row {i} | {i},000 | {i},500 |" for i in range(n_rows))
    return "|  | 2024 | 2023 |\n|---|---|---|\n" + rows


def test_count_tokens_splits_numbers_and_punctuation():
    # Revenue was $ 1 , 234 . 5 million .
    assert count_tokens("Revenue was $1,234.5 million.") == 10


# ---------- fixed-size ----------


def test_fixed_windows_overlap_and_cover_everything():
    filing = make_filing([Section("1", "Business", [Block("text", words(250))])])
    chunks = chunk_fixed(filing, max_tokens=100, overlap=20)
    assert [c.n_tokens for c in chunks] == [100, 100, 90]  # steps of 80: 0, 80, 160
    assert all(c.text == c.embed_text for c in chunks)  # baseline: no header
    assert chunks[0].strategy == "fixed-100-20"


def test_fixed_labels_section_where_chunk_starts():
    filing = make_filing(
        [
            Section("1", "Business", [Block("text", words(60))]),
            Section("1A", "Risk Factors", [Block("text", words(60, "beta"))]),
        ]
    )
    chunks = chunk_fixed(filing, max_tokens=50, overlap=0)
    assert [c.section_key for c in chunks] == ["1", "1", "1A"]
    assert "alpha" in chunks[1].text and "beta" in chunks[1].text  # crosses sections


def test_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_fixed(make_filing([]), max_tokens=10, overlap=10)


# ---------- section-aware ----------


def test_never_crosses_sections_and_adds_header():
    filing = make_filing(
        [
            Section("1", "Business", [Block("text", "We sell anvils.")]),
            Section("1A", "Risk Factors", [Block("text", "Coyotes are a risk.")]),
        ]
    )
    chunks = chunk_section_aware(filing, max_tokens=200)
    assert [(c.section_key, c.text) for c in chunks] == [
        ("1", "We sell anvils."),
        ("1A", "Coyotes are a risk."),
    ]
    assert chunks[1].embed_text.startswith("ACME Corp (ACME) | Form 10-K | FY2024")
    assert "Risk Factors\n\nCoyotes are a risk." in chunks[1].embed_text


def test_every_chunk_within_budget():
    blocks = [Block("text", words(130)), Block("table", table(80)), Block("text", words(900))]
    filing = make_filing([Section("7", "MD&A", blocks)])
    chunks = chunk_section_aware(filing, max_tokens=256)
    assert len(chunks) > 5
    assert all(c.n_tokens <= 256 for c in chunks)


def test_split_table_repeats_header_and_keeps_rows_whole():
    pieces = split_table(table(60), max_tokens=120)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith("|  | 2024 | 2023 |\n|---|---|---|\n")
        assert all(line.startswith("| ") or line == "|---|---|---|" for line in piece.split("\n"))


def test_lead_in_moves_with_its_table():
    blocks = [
        Block("text", words(150)),
        Block("text", "Net sales by category (in millions):"),
        Block("table", table(10)),
    ]
    chunks = chunk_section_aware(make_filing([Section("7", "MD&A", blocks)]), max_tokens=220)
    table_chunk = next(c for c in chunks if "| Row 0 |" in c.text)
    assert table_chunk.text.startswith("Net sales by category (in millions):")


def test_long_paragraph_split_on_sentences():
    text = " ".join(f"Sentence number {i} talks about revenue growth." for i in range(100))
    chunks = chunk_section_aware(
        make_filing([Section("7", "MD&A", [Block("text", text)])]), max_tokens=120
    )
    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)


def test_chunk_ids_unique_and_ordered():
    blocks = [Block("text", words(400))]
    chunks = chunk_section_aware(make_filing([Section("1", "Business", blocks)]), max_tokens=128)
    ids = [c.chunk_id for c in chunks]
    assert ids == sorted(set(ids))
    assert ids[0] == "0000000000-24-000001:section-128:0000"
