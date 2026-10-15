"""Baseline: fixed-size token windows with overlap, ignoring document structure.

This is the "naive RAG" strategy most tutorials use, kept deliberately simple
so the eval can show what structure-awareness buys us. Windows can start
mid-sentence and cut tables in half. The embedded text has no context header.
"""

from __future__ import annotations

import bisect

from finrag.chunking.base import TOKEN_RE, Chunk, Filing


def chunk_fixed(filing: Filing, max_tokens: int = 512, overlap: int = 64) -> list[Chunk]:
    if not 0 <= overlap < max_tokens:
        raise ValueError("need 0 <= overlap < max_tokens")
    strategy = f"fixed-{max_tokens}-{overlap}"

    # Concatenate all sections, remembering where each starts so every chunk
    # can still be labeled with the section it begins in (needed for evals).
    parts, starts, pos = [], [], 0
    for section in filing.sections:
        starts.append(pos)
        text = section.text
        parts.append(text)
        pos += len(text) + 2
    full = "\n\n".join(parts)

    spans = [m.span() for m in TOKEN_RE.finditer(full)]
    chunks = []
    step = max_tokens - overlap
    for n, first in enumerate(range(0, len(spans), step)):
        window = spans[first : first + max_tokens]
        start, end = window[0][0], window[-1][1]
        section = filing.sections[bisect.bisect_right(starts, start) - 1]
        text = full[start:end]
        chunks.append(Chunk.build(filing, strategy, n, section, text, text))
        if first + max_tokens >= len(spans):
            break
    return chunks
