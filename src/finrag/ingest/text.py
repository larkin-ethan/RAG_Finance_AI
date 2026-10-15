"""Text extraction helpers shared by the HTML and table parsers."""

from __future__ import annotations

import re

from bs4 import NavigableString, PageElement, Tag
from bs4.element import Comment, Declaration, Doctype, ProcessingInstruction

# Elements whose boundaries imply whitespace. Everything else (span, font, b,
# a, ix:nonfraction, ...) is inline: its text joins its neighbors directly.
# EDGAR often splits one number across spans: <span>1</span><span>,234</span>.
# Joining those with a space would produce "1 ,234".
SPACED_TAGS = {
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
    "table",
    "tr",
    "td",
    "th",
}
_SKIP = (Comment, Declaration, Doctype, ProcessingInstruction)


def is_text(node: PageElement) -> bool:
    return isinstance(node, NavigableString) and not isinstance(node, _SKIP)


def inline_text(node: PageElement) -> str:
    """Text of a node, adding spaces only where the markup implies a break."""
    if is_text(node):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    if node.name == "br":
        return " "
    inner = "".join(inline_text(child) for child in node.children)
    return f" {inner} " if node.name in SPACED_TAGS else inner


def clean_text(text: str) -> str:
    """Normalize whitespace (incl. non-breaking spaces) to single spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()
