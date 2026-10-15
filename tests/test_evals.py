import json

from finrag.evals.dataset import EvalQuestion, Gold, assign_stratified_split, load_questions
from finrag.evals.metrics import gold_coverage, numbers_correct, reciprocal_rank
from finrag.evals.report import table
from finrag.evals.run import EvalConfig, run_eval
from finrag.generate.llm import LLMResponse
from finrag.retrieve.vector import Hit

ACC = "0000320193-24-000123"
TABLE = "| | 2024 |\n|---|---|\n| Total net sales | $391,035 |"
REV = Gold(ACC, "AAPL", "391,035", value=391_035_000_000, unit="USD", fiscal_year=2024)


def hit(text: str, accession: str = ACC, n: int = 0) -> Hit:
    return Hit(f"{accession}:section-512:{n:04d}", 0.8, "AAPL", "Apple Inc.", "10-K", 2024,
               None, "2024-09-28", "7", "MD&A", "u", text)  # fmt: skip


# ---------- metrics ----------


def test_relevance_needs_right_filing_and_evidence():
    pairs = [("other-acc", TABLE), (ACC, "no number here"), (ACC, TABLE)]
    assert gold_coverage(pairs, [REV], k=2) == 0.0  # right number, wrong filing; then no number
    assert gold_coverage(pairs, [REV], k=3) == 1.0
    assert reciprocal_rank(pairs, [REV]) == 1 / 3


def test_multi_gold_coverage_is_fractional():
    other = Gold("acc-2023", "AAPL", "383,285", value=383_285e6)
    assert gold_coverage([(ACC, TABLE)], [REV, other], k=5) == 0.5


def test_numbers_correct_unit_aware():
    assert numbers_correct("Revenue was $391.0 billion.", [REV])
    assert numbers_correct("Revenue was $391,035 million.", [REV])
    assert not numbers_correct("Revenue was $383.3 billion.", [REV])
    eps = Gold(ACC, "AAPL", "6.08", value=6.08, unit="USD/shares")
    assert numbers_correct("Diluted EPS was $6.08.", [eps])
    assert not numbers_correct("Diluted EPS was $6.11.", [eps])


# ---------- dataset ----------


def test_split_is_frozen_and_stratified(tmp_path):
    qs = [EvalQuestion(f"q{i}", "numeric", "q?", True, "a") for i in range(10)]
    qs += [EvalQuestion(f"u{i}", "unanswerable", "q?", False, "idk") for i in range(10)]
    assign_stratified_split(qs)
    by_type = {
        t: sum(q.split == "test" for q in qs if q.type == t) for t in ("numeric", "unanswerable")
    }
    assert by_type == {"numeric": 3, "unanswerable": 3}

    path = tmp_path / "q.jsonl"
    from finrag.evals.dataset import save_questions

    save_questions(qs, path)
    reloaded = {q.id: q.split for q in load_questions(path)}
    assert reloaded == {q.id: q.split for q in qs}  # stored, not recomputed


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "q.jsonl"
    row = json.dumps({"id": "x", "type": "text", "question": "q", "answerable": True,
                      "reference_answer": "a"})  # fmt: skip
    path.write_text(row + "\n" + row + "\n")
    try:
        load_questions(path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ---------- runner ----------


class ScriptedLLM:
    """Generator + judge in one: answers from a script, judges always pass."""

    model = "scripted"

    def __init__(self, answer: dict):
        self.answer = answer

    def generate(self, system, prompt, schema=None, temperature=0.0):
        if "grade" in system or "check" in system:  # judge prompts
            return LLMResponse(json.dumps({"pass": True, "reason": "ok"}), self.model)
        return LLMResponse(json.dumps(self.answer), self.model, 1000, 100, 0.5)


def test_run_eval_end_to_end_with_fakes():
    questions = [
        EvalQuestion(
            "num-aapl",
            "numeric",
            "Apple FY2024 revenue?",
            True,
            "$391,035 million",
            gold=[REV],
            split="dev",
        ),
        EvalQuestion("unans-x", "unanswerable", "Tesla revenue?", False, "idk", split="dev"),
    ]
    llm = ScriptedLLM({
        "answerable": True,
        "answer": "Apple's total net sales were $391,035 million [2].",
        "citations": [{"source": 2, "quote": "Total net sales | $391,035"}],
    })  # fmt: skip
    retrieve = lambda q, k: [hit("filler"), hit(TABLE, n=1)]
    out = run_eval(questions, retrieve, EvalConfig("t"), llm, llm, prices=(0.5, 2.0))
    s = out["summary"]
    assert s["recall@1"] == 0.0 and s["recall@5"] == 1.0 and s["mrr"] == 0.5
    # The scripted model answers the Tesla question too: a failed abstention.
    assert s["correctness_by_type"] == {"numeric": 1.0, "unanswerable": 0.0}
    assert s["abstain_when_unanswerable"] == 0.0
    assert s["grounded_rate"] == 1.0
    assert s["cost_per_query_usd"] == (1000 * 0.5 + 100 * 2.0) / 1e6


def test_retrieval_mode_makes_no_llm_calls():
    questions = [EvalQuestion("num-aapl", "numeric", "q", True, "a", gold=[REV], split="dev")]
    out = run_eval(questions, lambda q, k: [hit(TABLE)], EvalConfig("r", mode="retrieval"))
    assert out["summary"]["recall@1"] == 1.0
    assert "answer_correctness" not in out["summary"]


def test_report_table_shows_change():
    base = {"recall@5": 0.80, "answer_correctness": 0.70, "latency_s_p50": 2.0}
    final = {"recall@5": 0.90, "answer_correctness": 0.82, "latency_s_p50": 2.5}
    md = table([base, final], ["baseline", "final"])
    assert "| Recall@5 | 80.0% | 90.0% | +10.0 pts ✅ |" in md
    assert "| Latency p50 (s) | 2.00 | 2.50 | +0.50s |" in md  # slower: no checkmark


def test_gate_flags_regressions_beyond_threshold():
    from finrag.evals.gate import compare

    ok, md = compare({"recall@5": 0.80, "mrr": 0.70}, {"recall@5": 0.81, "mrr": 0.70}, 0.02)
    assert ok and "✅" in md
    ok, md = compare({"recall@5": 0.75, "mrr": 0.70}, {"recall@5": 0.81, "mrr": 0.70}, 0.02)
    assert not ok and "❌ regression" in md
