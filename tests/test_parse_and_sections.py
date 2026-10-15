from pathlib import Path

import pytest

from finrag.ingest.models import Block
from finrag.ingest.parse_html import parse_blocks
from finrag.ingest.sections import PREAMBLE, split_sections

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def k_sections():
    blocks = parse_blocks((FIXTURES / "sample_10k.htm").read_bytes())
    return {s.key: s for s in split_sections(blocks, "10-K")}


@pytest.fixture(scope="module")
def q_sections():
    blocks = parse_blocks((FIXTURES / "sample_10q.htm").read_bytes())
    return {s.key: s for s in split_sections(blocks, "10-Q")}


def test_hidden_xbrl_and_page_furniture_removed():
    text = "\n".join(b.text for b in parse_blocks((FIXTURES / "sample_10k.htm").read_bytes()))
    assert "HIDDEN XBRL METADATA" not in text
    assert "2024 Form 10-K | 1" not in text
    assert "Table of Contents" not in text


def test_10k_items_found_in_order(k_sections):
    assert list(k_sections) == [PREAMBLE, "1", "1A", "4", "7", "16"]


def test_toc_only_item_is_dropped(k_sections):
    # Item 9B appears only in the table of contents.
    assert "9B" not in k_sections


def test_body_heading_wins_over_toc(k_sections):
    assert "designs, manufactures and markets anvils" in k_sections["1"].text
    assert "Item 1A. Risk Factors" in k_sections[PREAMBLE].text  # TOC stays in preamble


def test_split_heading_title_not_duplicated(k_sections):
    risk = k_sections["1A"]
    assert risk.title == "Risk Factors"
    assert risk.blocks[0].text.startswith("Demand for anvils")


def test_short_item_and_layout_table_heading(k_sections):
    assert k_sections["4"].text == "Not applicable."
    assert k_sections["16"].text == "None."


def test_part_markers_not_in_bodies(k_sections):
    assert "PART II" not in k_sections["4"].text


def test_financial_table_aligned(k_sections):
    table = next(b for b in k_sections["7"].blocks if b.kind == "table")
    lines = table.text.splitlines()
    assert lines[0] == "|  | 2024 | Change | 2023 |"
    assert "| Anvils | $201,183 | (2)% | $205,489 |" in lines
    assert "| Net loss on hedges | $(1,234) | n/m | $(900) |" in lines


def test_10q_keys_include_part(q_sections):
    assert list(q_sections) == [PREAMBLE, "I-1", "I-2", "II-1", "II-1A"]
    assert "W. E. Coyote" in q_sections["II-1"].text
    assert q_sections["II-1"].title == "Legal Proceedings"


def test_cross_reference_paragraph_is_not_a_heading():
    blocks = [
        Block("text", "Item 7. Management's Discussion"),
        Block("text", "Item 7 of this report discusses liquidity at length. " * 10),
        Block("text", "Revenue grew."),
    ]
    sections = split_sections(blocks, "10-K")
    assert [s.key for s in sections] == ["7"]


def test_no_headings_gives_single_preamble():
    sections = split_sections([Block("text", "Just some text.")], "10-K")
    assert [s.key for s in sections] == [PREAMBLE]


def test_numbers_split_across_spans_stay_intact():
    html = (
        "<p>Revenue was $<span>1</span><span>,234</span> million as of "
        "September <span>2</span><span>8</span>, 2024.<!-- a comment --></p>"
        "<p>Line one<br>line two</p>"
    )
    texts = [b.text for b in parse_blocks(html)]
    assert texts == [
        "Revenue was $1,234 million as of September 28, 2024.",
        "Line one line two",
    ]


def test_table_cell_spans_joined_without_spaces():
    html = (
        "<table><tr><td>Year</td><td><span>20</span><span>24</span></td></tr>"
        "<tr><td>Sales</td><td><span>1</span>,<span>234</span></td></tr></table>"
    )
    assert "| Sales | 1,234 |" in parse_blocks(html)[0].text


def test_10q_fallback_titles_when_item_headings_missing():
    # JPMorgan-style 10-Q: no "Item 1/2" in the body, MD&A before statements.
    blocks = [
        Block("text", "FORM 10-Q"),
        Block("text", "EXECUTIVE OVERVIEW"),
        Block("text", "Net income was $14.9 billion."),
        Block("text", "Consolidated statements of income (unaudited)"),
        Block("table", "| | 2026 |\n|---|---|\n| Revenue | 45,000 |"),
        Block("text", "Item 4. Controls and Procedures."),
        Block("text", "Disclosure controls are effective."),
    ]
    sections = {s.key: s for s in split_sections(blocks, "10-Q")}
    assert list(sections) == [PREAMBLE, "I-2", "I-1", "I-4"]
    assert sections["I-2"].text == "Net income was $14.9 billion."
    assert "Revenue" in sections["I-1"].text


def test_fallback_not_used_for_10k():
    blocks = [
        Block("text", "EXECUTIVE OVERVIEW"),
        Block("text", "Item 7. MD&A"),
        Block("text", "x"),
    ]
    assert [s.key for s in split_sections(blocks, "10-K")] == [PREAMBLE, "7"]


def test_10k_pointer_items_relocated_from_appended_annual_report():
    # JPMorgan style: Items 7/8 just point to pages; the report follows Item 15.
    filler = [Block("text", "Exhibit and report filler. " * 400) for _ in range(12)]
    blocks = [
        Block("text", "Item 7. Management's Discussion and Analysis"),
        Block("text", "MD&A appears on pages 46-160 of the Annual Report."),
        Block("text", "Item 8. Financial Statements and Supplementary Data"),
        Block("text", "The financial statements appear on pages 165-314."),
        Block("text", "Item 15. Exhibits and Financial Statement Schedules"),
        Block("text", "Exhibit index."),
        Block("text", "Management’s discussion and analysis"),
        Block("text", "Net income was $57.0 billion."),
        *filler,
        Block("text", "Management’s discussion and analysis"),  # running page header
        Block("text", "Report of Independent Registered Public Accounting Firm"),
        Block("table", "| | 2025 |\n|---|---|\n| Total assets | 4,424,900 |"),
    ]
    sections = split_sections(blocks, "10-K")
    by_key = {s.key: s for s in sections}
    assert [s.key for s in sections] == ["15", "7", "8"]  # relocated after the report
    assert "Net income was $57.0 billion." in by_key["7"].text
    assert "pages 46-160" in by_key["7"].text  # pointer kept as context
    assert "4,424,900" in by_key["8"].text and "4,424,900" not in by_key["7"].text
    assert by_key["15"].text.strip() == "Exhibit index."


def test_normal_10k_untouched_by_relocation():
    blocks = parse_blocks((FIXTURES / "sample_10k.htm").read_bytes())
    assert [s.key for s in split_sections(blocks, "10-K")] == [PREAMBLE, "1", "1A", "4", "7", "16"]
