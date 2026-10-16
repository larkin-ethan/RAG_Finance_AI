"""What changed in a company's Risk Factors (Item 1A) since last year's 10-K.

* Paragraphs are matched year over year by embedding similarity (cosine >=
  MATCH_THRESHOLD), so a lightly reworded risk counts as the same risk and
  genuinely new risks stand out. Exact matches skip the embedding step.
* new_share / removed_share: the share of this year's / last year's risk text
  (by characters) with no match on the other side.
* Red flags: specific phrases that are NEW this year (absent last year).
  Phrases, not words: nearly every 10-K mentions "covenants"; "not in
  compliance with ... covenants" is the signal.
* MD&A tone: change in the share of negative words (small curated list;
  the Loughran-McDonald dictionary needs a license for non-academic use).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from finrag.generate.citations import normalize

MATCH_THRESHOLD = 0.85
MIN_PARAGRAPH_WORDS = 25
RED_FLAGS = {
    "going concern": r"substantial doubt about (?:our|the company's) ability to continue as a going concern",
    "material weakness": r"material weakness(?:es)? in (?:our )?internal control",
    "restatement": r"\brestate(?:d|ment)\b.{0,60}financial statements",
    "government investigation": r"(?:subpoena|wells notice|under investigation by|investigation by the (?:sec|doj|department of justice))",
    "covenant breach": r"(?:not in compliance with|breach(?:ed)? of|waiver of).{0,60}covenants?",
    "default": r"(?:event of default has occurred|we (?:are|were) in default)",
    "delisting": r"(?:delisting|notice of non-?compliance).{0,60}(?:nasdaq|nyse|listing standards)",
}
NEGATIVE_WORDS = {
    "adverse", "adversely", "decline", "declined", "declines", "decrease", "decreased", "deteriorate",
    "deteriorated", "deterioration", "difficult", "disruption", "disruptions", "downturn", "failure",
    "impairment", "impairments", "litigation", "loss", "losses", "lower", "negative", "negatively",
    "penalties", "recession", "restructuring", "risk", "risks", "shortage", "shortfall", "slowdown",
    "uncertain", "uncertainty", "unfavorable", "volatile", "volatility", "weak", "weakness", "worse",
}  # fmt: skip

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class RiskDiff:
    new_share: float
    removed_share: float
    new_paragraphs: list[str] = field(default_factory=list)  # longest first, for the explanation
    red_flags: list[str] = field(default_factory=list)
    tone_change: float | None = None  # change in MD&A negative-word share (percentage points)


def paragraphs(section_text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", section_text)
    return [
        p.strip()
        for p in parts
        if len(p.split()) >= MIN_PARAGRAPH_WORDS and not p.lstrip().startswith("|")
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _unmatched(source: list[str], target: list[str], vec: dict[str, list[float]]) -> list[str]:
    target_norm = {normalize(t) for t in target}
    out = []
    for p in source:
        if normalize(p) in target_norm:
            continue
        if not any(_cosine(vec[p], vec[t]) >= MATCH_THRESHOLD for t in target):
            out.append(p)
    return out


def negative_share(text: str) -> float | None:
    words = re.findall(r"[a-z]+", text.lower())
    return sum(w in NEGATIVE_WORDS for w in words) / len(words) if words else None


def new_red_flags(current: str, prior: str) -> list[str]:
    cur, prev = normalize(current), normalize(prior)
    return [name for name, pattern in RED_FLAGS.items()
            if re.search(pattern, cur) and not re.search(pattern, prev)]  # fmt: skip


def diff_risk_factors(current_1a: str, prior_1a: str, embed: EmbedFn, current_mdna: str = "",
                      prior_mdna: str = "") -> RiskDiff:  # fmt: skip
    cur, prev = paragraphs(current_1a), paragraphs(prior_1a)
    unique = list(dict.fromkeys(cur + prev))
    vec = dict(zip(unique, embed(unique))) if unique else {}
    new = _unmatched(cur, prev, vec)
    removed = _unmatched(prev, cur, vec)
    total_cur, total_prev = sum(map(len, cur)) or 1, sum(map(len, prev)) or 1
    tone = None
    if current_mdna and prior_mdna:
        a, b = negative_share(current_mdna), negative_share(prior_mdna)
        tone = None if a is None or b is None else round(100 * (a - b), 3)
    return RiskDiff(
        new_share=round(sum(map(len, new)) / total_cur, 4),
        removed_share=round(sum(map(len, removed)) / total_prev, 4),
        new_paragraphs=sorted(new, key=len, reverse=True)[:5],
        red_flags=new_red_flags(current_1a, prior_1a),
        tone_change=tone,
    )
