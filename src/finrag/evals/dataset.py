"""Eval question format, loading, and the dev/test split.

Gold evidence is chunking-strategy independent: a retrieved chunk is relevant
if it comes from the gold filing (accession_number) and contains the evidence
string (e.g. "391,035"). That keeps recall@k comparable across strategies,
whose chunk IDs differ.

Leakage note: `tickers`/`fiscal_years` describe the question for analysis.
The system under test must NOT receive them as filters; at eval time any
filters have to be extracted from the question text itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

QuestionType = Literal["numeric", "multi_filing", "comparison", "text", "unanswerable"]
Split = Literal["dev", "test"]
TEST_FRACTION = 0.3


@dataclass
class Gold:
    accession_number: str
    ticker: str
    evidence: str  # substring a relevant chunk must contain (normalized match)
    value: float | None = None  # numeric answer, if any (in base units, e.g. USD)
    unit: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    concept: str | None = None  # XBRL tag the value came from


@dataclass
class EvalQuestion:
    id: str
    type: QuestionType
    question: str
    answerable: bool
    reference_answer: str
    gold: list[Gold] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    # xbrl: generated from SEC XBRL facts; template: unanswerable templates;
    # human: written by the project owner; assistant: drafted by Claude, needs review.
    source: Literal["xbrl", "human", "template", "assistant"] = "human"
    reviewed: bool = False
    # Frozen once assigned and stored in the file: a question must never move
    # from test to dev after we've tuned on dev (that would leak). If missing
    # (e.g. a newly hand-written question), it falls back to a stable hash.
    split: Split | None = None

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = "test" if hash_bucket(self.id) < TEST_FRACTION else "dev"

    @classmethod
    def from_dict(cls, d: dict) -> EvalQuestion:
        return cls(**(d | {"gold": [Gold(**g) for g in d.get("gold", [])]}))


def hash_bucket(key: str) -> float:
    """Stable pseudo-random number in [0, 1) derived from a string."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 16**8


def assign_stratified_split(questions: list[EvalQuestion]) -> None:
    """Within each question type, the ~30% with the lowest hash go to test.
    Balanced across types even for small sets, and deterministic."""
    by_type: dict[str, list[EvalQuestion]] = {}
    for q in questions:
        by_type.setdefault(q.type, []).append(q)
    for group in by_type.values():
        group.sort(key=lambda q: hash_bucket(q.id))
        n_test = round(len(group) * TEST_FRACTION)
        for i, q in enumerate(group):
            q.split = "test" if i < n_test else "dev"


def load_questions(*paths: Path, split: str | None = None) -> list[EvalQuestion]:
    questions = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("//"):
                questions.append(EvalQuestion.from_dict(json.loads(line)))
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate question ids: {dupes}")
    return [q for q in questions if split is None or q.split == split]


def save_questions(questions: list[EvalQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
