"""CLI: compare eval runs as a markdown table (for the README results table).

python -m finrag.evals.report evals/results/baseline-dev.json evals/results/final-dev.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    ("recall@1", "Recall@1", "pct", True),
    ("recall@5", "Recall@5", "pct", True),
    ("recall@10", "Recall@10", "pct", True),
    ("mrr", "MRR", "num", True),
    ("answer_correctness", "Answer correctness", "pct", True),
    ("faithfulness", "Faithfulness (judge)", "pct", True),
    ("grounded_rate", "Citations verified (deterministic)", "pct", True),
    ("abstain_when_unanswerable", 'Says "I don\'t know" when it should', "pct", True),
    ("false_abstention_rate", 'False "I don\'t know" rate', "pct", False),
    ("latency_s_p50", "Latency p50 (s)", "sec", False),
    ("latency_s_p95", "Latency p95 (s)", "sec", False),
    ("cost_per_query_usd", "Cost per query (USD, list price)", "usd", False),
]


def fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    return {"pct": f"{value:.1%}", "num": f"{value:.3f}", "sec": f"{value:.2f}",
            "usd": f"${value:.4f}"}[kind]  # fmt: skip


def delta(first, last, kind: str, higher_better: bool) -> str:
    if first is None or last is None:
        return ""
    d = last - first
    better = (d > 0) == higher_better
    sign = "+" if d >= 0 else "−"
    body = {"pct": f"{abs(d) * 100:.1f} pts", "num": f"{abs(d):.3f}", "sec": f"{abs(d):.2f}s",
            "usd": f"${abs(d):.4f}"}[kind]  # fmt: skip
    return f"{sign}{body}" + (" ✅" if better and d else "")


def table(summaries: list[dict], names: list[str]) -> str:
    header = "| Metric | " + " | ".join(names) + (" | Change |" if len(names) > 1 else " |")
    lines = [header, "|---" * (len(names) + 1 + (len(names) > 1)) + "|"]
    for key, label, kind, higher_better in ROWS:
        values = [s.get(key) for s in summaries]
        if all(v is None for v in values):
            continue
        cells = [fmt(v, kind) for v in values]
        change = [delta(values[0], values[-1], kind, higher_better)] if len(names) > 1 else []
        lines.append(f"| {label} | " + " | ".join(cells + change) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    data = [json.loads(p.read_text(encoding="utf-8")) for p in args.results]
    names = [d["summary"]["config"]["name"] for d in data]
    n = data[0]["summary"]["n"]
    print(f"Eval results ({n} questions)\n")
    print(table([d["summary"] for d in data], names))


if __name__ == "__main__":
    main()
