"""Run the eval: retrieval metrics, answer correctness, faithfulness, abstention,
latency and cost. Writes evals/results/{name}-{split}.json.

  python -m finrag.evals.run --name baseline --split dev                # full eval
  python -m finrag.evals.run --name baseline --split dev --mode retrieval  # no LLM (CI)
  python -m finrag.evals.run --name hybrid-auto-rerank --method hybrid --filters auto --rerank

The system under test gets ONLY the question text: no gold tickers or years
as filters (that would leak the answer location). LLM calls are cached, so an
interrupted run (e.g. a free-tier daily quota hit) resumes for free.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from finrag.evals.dataset import EvalQuestion, load_questions
from finrag.evals.judge import judge_correctness, judge_faithfulness
from finrag.evals.metrics import gold_coverage, mean, numbers_correct, reciprocal_rank
from finrag.generate.answer import answer_question
from finrag.generate.llm import LLMClient
from finrag.retrieve.vector import Hit

QUESTION_FILES = [
    Path("evals/questions/generated.jsonl"),
    Path("evals/questions/human.jsonl"),
    Path("evals/questions/assistant_drafted.jsonl"),
]
RECALL_KS = (1, 3, 5, 10)


@dataclass
class EvalConfig:
    name: str
    strategy: str = "section-512"
    k: int = 6  # passages given to the LLM
    retrieve_k: int = 10  # passages retrieved (for recall@10)
    mode: Literal["retrieval", "full"] = "full"
    method: Literal["vector", "keyword", "hybrid"] = "vector"
    filters: Literal["none", "auto"] = "none"
    rerank: bool = False


@dataclass
class QuestionResult:
    id: str
    type: str
    split: str
    question: str
    recall: dict[int, float] = field(default_factory=dict)
    rr: float | None = None
    retrieved: list[dict] = field(default_factory=list)
    filters: dict = field(default_factory=dict)  # what the parser extracted (auto mode)
    retrieval_s: float = 0.0
    answer: str | None = None
    abstained: bool | None = None
    grounded: bool | None = None
    unsupported_numbers: list[str] = field(default_factory=list)
    correct: bool | None = None
    correct_method: str | None = None
    correct_reason: str = ""
    faithful: bool | None = None
    faithful_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    generation_s: float = 0.0


Retriever = Callable[[str, int], list[Hit]]


def evaluate_question(
    q: EvalQuestion, retrieve: Retriever, cfg: EvalConfig,
    llm: LLMClient | None, judge: LLMClient | None,
) -> QuestionResult:  # fmt: skip
    r = QuestionResult(q.id, q.type, q.split or "", q.question)
    started = time.time()
    hits = retrieve(q.question, cfg.retrieve_k)
    r.retrieval_s = time.time() - started
    if hasattr(retrieve, "last_filters"):
        r.filters = {k: v for k, v in asdict(retrieve.last_filters).items() if v}
    r.retrieved = [
        {"chunk_id": h.chunk_id, "ticker": h.ticker, "form": h.form, "fiscal_year": h.fiscal_year,
         "section": h.section_key, "score": round(h.score, 4)}
        for h in hits
    ]  # fmt: skip
    if q.answerable and q.gold:
        pairs = [(h.chunk_id.split(":")[0], h.text) for h in hits]
        r.recall = {k: gold_coverage(pairs, q.gold, k) for k in RECALL_KS}
        r.rr = reciprocal_rank(pairs, q.gold)

    if cfg.mode == "retrieval" or llm is None:
        return r

    context = hits[: cfg.k]
    ans = answer_question(q.question, lambda _: context, llm)
    r.answer, r.abstained, r.grounded = ans.text, not ans.answerable, ans.grounded
    r.unsupported_numbers = ans.unsupported_numbers
    r.input_tokens, r.output_tokens, r.generation_s = (
        ans.input_tokens, ans.output_tokens, ans.generation_s,
    )  # fmt: skip

    # Correctness
    if not q.answerable:
        r.correct, r.correct_method = r.abstained, "abstention"
    elif r.abstained:
        r.correct, r.correct_method = False, "false_abstention"
    elif q.type in ("numeric", "multi_filing"):
        r.correct, r.correct_method = numbers_correct(ans.text, q.gold), "numeric"
    elif judge is not None:
        v = judge_correctness(judge, q.question, q.reference_answer, ans.text)
        numbers_ok = q.type != "comparison" or numbers_correct(ans.text, q.gold)
        r.correct, r.correct_method, r.correct_reason = v.passed and numbers_ok, "judge", v.reason

    # Faithfulness: only meaningful when the system actually answered
    if not r.abstained and judge is not None:
        cited = [c.hit.text for c in ans.citations if c.hit] or [h.text for h in context]
        v = judge_faithfulness(judge, ans.text, cited)
        r.faithful, r.faithful_reason = v.passed, v.reason
    return r


def summarize(results: list[QuestionResult], cfg: EvalConfig, prices: tuple | None) -> dict:
    answerable = [r for r in results if r.recall]
    summary: dict = {"config": asdict(cfg), "n": len(results)}
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r.type] += 1
    summary["n_by_type"] = dict(counts)
    for k in RECALL_KS:
        summary[f"recall@{k}"] = mean([r.recall[k] for r in answerable])
    summary["mrr"] = mean([r.rr for r in answerable if r.rr is not None])
    summary["retrieval_s_p50"] = (
        statistics.median([r.retrieval_s for r in results]) if results else None
    )

    judged = [r for r in results if r.correct is not None]
    if judged:
        summary["answer_correctness"] = mean([float(r.correct) for r in judged])
        by_type: dict[str, list[float]] = defaultdict(list)
        for r in judged:
            by_type[r.type].append(float(r.correct))
        summary["correctness_by_type"] = {t: mean(v) for t, v in sorted(by_type.items())}
        answered = [r for r in results if r.abstained is False]
        summary["faithfulness"] = mean(
            [float(r.faithful) for r in answered if r.faithful is not None]
        )
        summary["grounded_rate"] = mean([float(bool(r.grounded)) for r in answered])
        unans = [r for r in results if r.type == "unanswerable"]
        ans_q = [r for r in results if r.type != "unanswerable" and r.abstained is not None]
        summary["abstain_when_unanswerable"] = mean([float(bool(r.abstained)) for r in unans])
        summary["false_abstention_rate"] = mean([float(bool(r.abstained)) for r in ans_q])
        totals = [r.retrieval_s + r.generation_s for r in results if r.abstained is not None]
        summary["latency_s_p50"] = statistics.median(totals) if totals else None
        summary["latency_s_p95"] = sorted(totals)[int(0.95 * (len(totals) - 1))] if totals else None
        summary["tokens_in_mean"] = mean([r.input_tokens for r in judged])
        summary["tokens_out_mean"] = mean([r.output_tokens for r in judged])
        if prices and all(p is not None for p in prices):
            p_in, p_out = prices
            summary["cost_per_query_usd"] = (
                summary["tokens_in_mean"] * p_in + summary["tokens_out_mean"] * p_out
            ) / 1e6
    return summary


def run_eval(
    questions: list[EvalQuestion], retrieve: Retriever, cfg: EvalConfig,
    llm: LLMClient | None = None, judge: LLMClient | None = None,
    prices: tuple | None = None, progress: bool = False,
) -> dict:  # fmt: skip
    results = []
    for i, q in enumerate(questions, 1):
        results.append(evaluate_question(q, retrieve, cfg, llm, judge))
        if progress:
            print(f"  {i}/{len(questions)} {q.id}", end="\r", flush=True)
    return {"summary": summarize(results, cfg, prices), "results": [asdict(r) for r in results]}


def main() -> None:
    from finrag.embed.embedder import Embedder
    from finrag.generate.llm import make_llm
    from finrag.retrieve.retriever import RetrievalConfig, db_retriever
    from finrag.settings import get_settings, load_watchlist
    from finrag.store.db import connect

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="label for this run, e.g. baseline")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--strategy", default="section-512")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--mode", choices=["retrieval", "full"], default="full")
    parser.add_argument("--method", choices=["vector", "keyword", "hybrid"], default="vector")
    parser.add_argument("--filters", choices=["none", "auto"], default="none")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--types", nargs="*")
    args = parser.parse_args()

    settings = get_settings()
    questions = load_questions(*QUESTION_FILES, split=args.split)
    if args.types:
        questions = [q for q in questions if q.type in args.types]
    questions = questions[: args.limit] if args.limit else questions
    cfg = EvalConfig(
        args.name, args.strategy, args.k, max(10, args.k), args.mode,
        args.method, args.filters, args.rerank,
    )  # fmt: skip
    retrieve = db_retriever(
        connect(settings.database_url),
        Embedder.from_settings(settings),
        RetrievalConfig(cfg.strategy, cfg.method, cfg.filters, cfg.rerank),
        load_watchlist(settings.watchlist_path).tickers,
    )

    full = args.mode == "full"
    llm = make_llm(settings) if full else None
    judge = make_llm(settings, settings.gemini_judge_model) if full else None
    prices = (settings.gemini_price_input_per_mtok, settings.gemini_price_output_per_mtok)

    print(f"Eval '{cfg.name}' on {len(questions)} {args.split} questions ({cfg.mode})")
    out = run_eval(questions, retrieve, cfg, llm, judge, prices, progress=True)
    path = Path("evals/results") / f"{cfg.name}-{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nWrote {path}")
    for key, value in out["summary"].items():
        if key not in ("config", "results"):
            print(f"  {key:28} {value:.3f}" if isinstance(value, float) else f"  {key:28} {value}")


if __name__ == "__main__":
    main()
