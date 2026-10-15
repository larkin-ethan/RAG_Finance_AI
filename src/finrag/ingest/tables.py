"""Convert EDGAR HTML tables to markdown.

EDGAR financial tables are layout-heavy. A single number like "$ (1,234)" is
often spread across three cells: "$" | "(1,234" | ")". Spacer columns are
empty in every row, and both headers and data cells use colspan. We:
  1. glue "$" onto the following cell and ")" / "%" / footnote letters like
     "(c)" onto the preceding one, on the original cells (before colspan
     expansion). A stray "(c)" in a spacer column otherwise blocks column
     merging and shifts every later column under the wrong header. Digit
     markers like "(1)" are left alone: in a financial table that's usually
     negative one.
  2. expand colspan by copying the cell across every spanned column, and
     remember which original cell each copy came from,
  3. drop rows and columns that are empty everywhere,
  4. merge adjacent columns whose non-empty cells are copies of the same
     original cell. A header like "2024" spanning [ $ | 1,234 | ) ] must end
     up directly above "$1,234", or the model reads the number under the
     wrong year. And a data cell "(2)" spanning two columns must appear once,
     not twice.
Tables that collapse to a single column/row are layout tables (headings,
bullet lists) and are returned as plain text instead.
"""

from __future__ import annotations

import re
from itertools import count

from bs4 import Tag

from finrag.ingest.models import Block
from finrag.ingest.text import clean_text, inline_text

_PREFIXES = {"$", "€", "£"}
_SUFFIXES = {")", "%", ")%", "%)"}
_FOOTNOTE = re.compile(r"^(\([a-z]{1,2}\))+$")

# A grid cell: (text, id of the original <td> it was copied from).
Cell = tuple[str, int]


def _row_cells(tr: Tag) -> list[list]:
    """Original cells as [text, colspan] pairs (mutable, for symbol merging)."""
    cells = []
    for cell in tr.find_all(["td", "th"], recursive=False):
        try:
            span = max(int(cell.get("colspan", 1)), 1)
        except ValueError:
            span = 1
        cells.append([clean_text(inline_text(cell)), span])
    return cells


def _merge_symbols(cells: list[list]) -> None:
    for i, (value, _) in enumerate(cells):
        if value in _PREFIXES:
            nxt = next((j for j in range(i + 1, len(cells)) if cells[j][0]), None)
            if nxt is not None:
                cells[nxt][0] = value + cells[nxt][0]
                cells[i][0] = ""
        elif value in _SUFFIXES or _FOOTNOTE.match(value):
            prev = next((j for j in range(i - 1, -1, -1) if cells[j][0]), None)
            if prev is not None:
                cells[prev][0] = cells[prev][0] + value
                cells[i][0] = ""


def _expand(cells: list[list], ids: count) -> list[Cell]:
    row: list[Cell] = []
    for text, span in cells:
        cell_id = next(ids)
        row.extend([(text, cell_id)] * span)
    return row


def _same_span(a: list[Cell], b: list[Cell]) -> bool:
    """Adjacent columns a and b belong together if, in every row, at most one
    is non-empty or both are copies of the same original cell, and at least
    one row shows such a copy."""
    shared = False
    for (x, x_id), (y, y_id) in zip(a, b):
        if x and y:
            if x_id != y_id:
                return False
            shared = True
    return shared


def table_rows(table: Tag) -> list[list[str]]:
    """Return the table as a clean rectangular grid of cell strings."""
    ids = count()
    rows = []
    for tr in table.find_all("tr"):
        cells = _row_cells(tr)
        _merge_symbols(cells)
        rows.append(_expand(cells, ids))
    rows = [r for r in rows if any(text for text, _ in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [("", -1)] * (width - len(r)) for r in rows]

    columns = [[r[c] for r in rows] for c in range(width) if any(r[c][0] for r in rows)]
    merged: list[list[Cell]] = []
    for col in columns:
        if merged and _same_span(merged[-1], col):
            merged[-1] = [x if x[0] else y for x, y in zip(merged[-1], col)]
        else:
            merged.append(col)
    return [[text for text, _ in row] for row in zip(*merged)]


def to_markdown(rows: list[list[str]]) -> str:
    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in row) + " |"

    header, *body = rows
    lines = [fmt(header), "|" + "---|" * len(header)]
    lines += [fmt(r) for r in body]
    return "\n".join(lines)


def table_to_block(table: Tag) -> Block | None:
    rows = table_rows(table)
    if not rows:
        return None
    if len(rows) == 1 or len(rows[0]) == 1:
        # Layout table: join into text, one line per row.
        text = "\n".join(" ".join(c for c in r if c) for r in rows)
        return Block("text", text)
    return Block("table", to_markdown(rows))
