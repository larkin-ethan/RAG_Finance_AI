from bs4 import BeautifulSoup

from finrag.ingest.tables import table_rows, table_to_block


def _table(html: str):
    return BeautifulSoup(html, "lxml").find("table")


def test_merges_currency_and_parentheses_and_drops_spacers():
    rows = table_rows(
        _table(
            "<table>"
            "<tr><td></td><td colspan='3'>2024</td><td></td><td colspan='3'>2023</td></tr>"
            "<tr><td>Loss</td><td>$</td><td>(1,234</td><td>)</td>"
            "<td></td><td>$</td><td>(900</td><td>)</td></tr>"
            "</table>"
        )
    )
    assert rows == [["", "2024", "2023"], ["Loss", "$(1,234)", "$(900)"]]


def test_spanning_header_does_not_merge_distinct_subcolumns():
    rows = table_rows(
        _table(
            "<table>"
            "<tr><td></td><td colspan='2'>Year Ended</td></tr>"
            "<tr><td></td><td>2024</td><td>2023</td></tr>"
            "<tr><td>Sales</td><td>100</td><td>90</td></tr>"
            "</table>"
        )
    )
    assert rows == [
        ["", "Year Ended", "Year Ended"],
        ["", "2024", "2023"],
        ["Sales", "100", "90"],
    ]


def test_percent_suffix_glued():
    rows = table_rows(_table("<table><tr><td>A</td><td>13</td><td>%</td></tr></table>"))
    assert rows == [["A", "13%"]]


def test_layout_table_becomes_text():
    block = table_to_block(
        _table("<table><tr><td>Item 16.</td><td>Form 10-K Summary</td></tr></table>")
    )
    assert block.kind == "text"
    assert block.text == "Item 16. Form 10-K Summary"


def test_empty_table_returns_none():
    assert table_to_block(_table("<table><tr><td> </td></tr></table>")) is None


def test_pipe_escaped_in_markdown():
    block = table_to_block(
        _table("<table><tr><td>a|b</td><td>x</td></tr><tr><td>c</td><td>1</td></tr></table>")
    )
    assert "a\\|b" in block.text


def test_spanned_data_cell_not_duplicated():
    # Apple 10-K FY2024 net sales table: "(2)" spans 2 columns, "%" separate.
    rows = table_rows(
        _table(
            "<table>"
            "<tr><td colspan='3'></td><td colspan='3'>2024</td><td colspan='3'></td>"
            "<td colspan='3'>Change</td></tr>"
            "<tr><td colspan='3'>iPhone</td><td>$</td><td>201,183</td><td></td>"
            "<td colspan='3'></td><td colspan='2'>(2)</td><td>%</td></tr>"
            "</table>"
        )
    )
    assert rows == [["", "2024", "Change"], ["iPhone", "$201,183", "(2)%"]]


def test_footnote_marker_in_spacer_glued_not_misaligning():
    # JPMorgan 10-Q: "(c)" sits in a spacer cell between the 2023 and 2022 columns.
    header = (
        "<tr><td colspan='3'>(in millions)</td><td colspan='3'>2023</td><td colspan='3'></td>"
        "<td colspan='3'>2022</td></tr>"
    )
    row_a = (
        "<tr><td colspan='3'>Securities</td><td></td><td>$195,200</td><td></td>"
        "<td colspan='3'>(c)</td><td></td><td>$186,441</td><td></td></tr>"
    )
    row_b = (
        "<tr><td colspan='3'>Held-to-maturity</td><td colspan='2'>388,261</td><td></td>"
        "<td colspan='3'></td><td colspan='2'>430,106</td><td></td></tr>"
    )
    rows = table_rows(_table(f"<table>{header}{row_a}{row_b}</table>"))
    assert rows == [
        ["(in millions)", "2023", "2022"],
        ["Securities", "$195,200(c)", "$186,441"],
        ["Held-to-maturity", "388,261", "430,106"],
    ]


def test_digit_in_parentheses_is_a_value_not_a_footnote():
    rows = table_rows(
        _table("<table><tr><td>A</td><td>2024</td></tr><tr><td>B</td><td>(1)</td></tr></table>")
    )
    assert rows == [["A", "2024"], ["B", "(1)"]]
