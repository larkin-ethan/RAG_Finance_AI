"""Answer a question: retrieve -> prompt -> LLM (JSON) -> verify citations.

Strict mode (default): if the model claims an answer but none of its citations
verify, we return "I don't know" instead. An unverifiable answer is exactly
the hallucination risk this project is about. Partially verified answers are
returned with `grounded=False` so the UI can warn.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from finrag.generate.citations import Verification, verify
from finrag.generate.llm import LLMClient, LLMError
from finrag.generate.prompts import ANSWER_SCHEMA, SYSTEM, build_prompt
from finrag.retrieve.vector import Hit

IDK = "I don't know. The retrieved filings don't contain enough information to answer that."


@dataclass
class Citation:
    number: int  # the [n] used in the answer text
    quote: str
    verified: bool
    hit: Hit | None


@dataclass
class Answer:
    question: str
    text: str
    answerable: bool
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    cached: bool = False
    abstain_reason: str = ""  # why we answered "I don't know", if we did


def _idk(question: str, hits: list[Hit], reason: str, **extra) -> Answer:
    return Answer(question, IDK, answerable=False, grounded=True, hits=hits,
                  abstain_reason=reason, **extra)  # fmt: skip


def answer_question(
    question: str,
    retrieve: Callable[[str], list[Hit]],
    llm: LLMClient,
    strict: bool = True,
) -> Answer:
    started = time.time()
    hits = retrieve(question)
    retrieval_s = time.time() - started
    if not hits:
        return _idk(question, hits, "no passages retrieved", retrieval_s=retrieval_s)

    try:
        response = llm.generate(SYSTEM, build_prompt(question, hits), ANSWER_SCHEMA)
        data = response.json()
    except (LLMError, json.JSONDecodeError, ValueError) as exc:
        return _idk(question, hits, f"LLM error: {exc}", retrieval_s=retrieval_s)

    usage = {
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "retrieval_s": retrieval_s,
        "generation_s": response.latency_s,
        "cached": response.cached,
    }
    if not data.get("answerable"):
        return _idk(question, hits, "model: not answerable from sources", **usage)

    text = str(data.get("answer", "")).strip()
    raw_citations = data.get("citations") or []
    check: Verification = verify(text, raw_citations, [h.text for h in hits])
    if strict and not check.verified_sources:
        return _idk(question, hits, "no citation could be verified", **usage)

    citations = [
        Citation(
            number=c.source,
            quote=c.quote,
            verified=c.valid_source and c.quote_found,
            hit=hits[c.source - 1] if c.valid_source else None,
        )
        for c in check.citations
    ]
    return Answer(
        question=question,
        text=text,
        answerable=True,
        grounded=check.grounded,
        citations=citations,
        unsupported_numbers=check.unsupported_numbers,
        hits=hits,
        **usage,
    )
