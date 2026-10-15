"""Turn a raw EDGAR filing (HTML / inline XBRL) into an ordered list of Blocks.

Walk strategy: block-level elements (p, div, h1...) can contain both inline
text and nested blocks. We accumulate inline runs and flush them as a text
Block whenever a nested block (or the element's end) is reached. Tables are
handed to tables.py whole.
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from finrag.ingest.models import Block
from finrag.ingest.tables import table_to_block
from finrag.ingest.text import clean_text, inline_text, is_text

# Inline XBRL filings are XHTML and may start with <?xml ...?>. The lenient
# HTML parser handles them better than a strict XML parser would.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = {
    "p",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ul",
    "ol",
    "section",
    "article",
    "center",
    "blockquote",
    "body",
    "html",
}
DROP_TAGS = ["script", "style", "head", "ix:header"]

# Page furniture repeated on every page: bare page numbers, back-links,
# and footers like "Apple Inc. | 2024 Form 10-K | 12".
_NOISE = re.compile(
    r"^(\d{1,3}|[ivxlc]{1,6}|table of contents|.{0,80}\|\s*\d{1,3})$", re.IGNORECASE
)


def _is_hidden(tag: Tag) -> bool:
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def _walk(el: Tag, out: list[Block]) -> None:
    buffer: list[str] = []

    def flush() -> None:
        text = clean_text("".join(buffer))
        buffer.clear()
        if text and not _NOISE.match(text):
            out.append(Block("text", text))

    for child in el.children:
        if is_text(child):
            buffer.append(str(child))
        elif isinstance(child, Tag):
            if child.name == "table":
                flush()
                block = table_to_block(child)
                if block and not (block.kind == "text" and _NOISE.match(block.text)):
                    out.append(block)
            elif child.name in BLOCK_TAGS:
                flush()
                _walk(child, out)
            elif child.name == "br":
                buffer.append(" ")
            else:  # inline: span, font, b, a, ix:nonfraction, ...
                buffer.append(inline_text(child))
    flush()


def parse_blocks(html: str | bytes) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(_is_hidden):
        tag.decompose()
    blocks: list[Block] = []
    _walk(soup.body or soup, blocks)
    return blocks
