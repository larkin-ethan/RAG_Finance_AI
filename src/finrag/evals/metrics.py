"""Eval metrics. Pure functions: no I/O, no LLM. Unit-tested.

Relevance is strategy independent: a retrieved chunk is relevant to a gold
item if it comes from the gold filing and contains the gold evidence string
(after normalization). See dataset.py.
"""

from __future__ import annotations

from finrag.evals.dataset import Gold
from finrag.generate.citations import extract_numbers, normalize


def is_relevant(accession_number: str, text: str, gold: Gold) -> bool:
    return accession_number == gold.accession_number and normalize(gold.evidence) in normalize(text)


def gold_coverage(hits: list[tuple[str, str]], golds: list[Gold], k: int) -> float:
    """Fraction of gold items with at least one relevant chunk in the top k.
    (= recall@k; for single-gold questions it's 0 or 1.) hits: (accession, text)."""
    if not golds:
        return 0.0
    top = hits[:k]
    covered = sum(any(is_relevant(a, t, g) for a, t in top) for g in golds)
    return covered / len(golds)


def reciprocal_rank(hits: list[tuple[str, str]], golds: list[Gold]) -> float:
    """1/rank of the first chunk relevant to any gold item (0 if none)."""
    for rank, (a, t) in enumerate(hits, 1):
        if any(is_relevant(a, t, g) for g in golds):
            return 1.0 / rank
    return 0.0


def numbers_correct(answer: str, golds: list[Gold], rel_tol: float = 0.001) -> bool:
    """Every gold value appears in the answer (unit-aware, rounding-tolerant).
    "$391.0 billion" and "$391,035 million" both match 391,035,000,000.
    Tolerance comes from the precision the answer states ("$391 billion" may
    be off by $0.5B). The small rel_tol only absorbs float noise; a looser one
    would accept EPS of $6.11 for $6.08."""
    found = extract_numbers(answer)
    for g in golds:
        if g.value is None:
            continue
        target = abs(g.value)
        if not any(abs(q.value - target) <= max(q.tolerance, rel_tol * target) for q in found):
            return False
    return True


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
