"""Data structures shared by the parsing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Block:
    """One paragraph-like unit of a filing, in document order."""

    kind: Literal["text", "table"]
    text: str  # plain text, or a markdown table for kind="table"


@dataclass
class Section:
    """A filing Item, e.g. key="1A" (10-K) or key="II-1A" (10-Q)."""

    key: str
    title: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)
