"""LLM-as-judge for what can't be checked deterministically.

The judge uses a different model (Flash-Lite) from the generator (Flash) to
reduce self-preference bias, runs at temperature 0, and returns a structured
verdict with a reason. Spot-check its verdicts by hand (evals/README.md) and
report the agreement rate; a judge is a measurement instrument and needs calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from finrag.generate.llm import LLMClient

CORRECTNESS_SYSTEM = """\
You grade answers to questions about SEC filings against a reference answer.
An answer is CORRECT if it conveys the same key facts as the reference (same
numbers within rounding, same entity, same period, same conclusion). Extra
correct detail is fine. Missing or wrong key facts make it incorrect.
Formatting differences ($391.0 billion vs $391,035 million) do not matter.
"""

FAITHFULNESS_SYSTEM = """\
You check whether an answer is fully supported by the source passages it was
given. FAITHFUL means every factual claim in the answer is stated in, or
directly follows from, the passages. Any claim not supported by the passages
(even if true in the real world) makes it unfaithful.
"""

VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"pass": {"type": "BOOLEAN"}, "reason": {"type": "STRING"}},
    "required": ["pass", "reason"],
    "propertyOrdering": ["reason", "pass"],  # reason first: think, then decide
}


@dataclass
class Verdict:
    passed: bool
    reason: str


def _ask(llm: LLMClient, system: str, prompt: str) -> Verdict:
    data = llm.generate(system, prompt, VERDICT_SCHEMA).json()
    return Verdict(bool(data.get("pass")), str(data.get("reason", "")))


def judge_correctness(llm: LLMClient, question: str, reference: str, answer: str) -> Verdict:
    prompt = f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade: {answer}"
    return _ask(llm, CORRECTNESS_SYSTEM, prompt)


def judge_faithfulness(llm: LLMClient, answer: str, passages: list[str]) -> Verdict:
    sources = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    prompt = f"Passages:\n\n{sources}\n\nAnswer to check: {answer}"
    return _ask(llm, FAITHFULNESS_SYSTEM, prompt)
