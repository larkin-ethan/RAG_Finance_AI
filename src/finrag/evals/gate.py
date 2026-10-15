"""CI gate: fail if retrieval quality regressed vs the committed baseline.

  python -m finrag.evals.gate evals/results/ci-dev.json evals/baseline_retrieval.json

Exit 1 if recall@5 or MRR dropped by more than --max-drop (default 0.02 = 2
points). Writes a markdown table to $GITHUB_STEP_SUMMARY when running in
GitHub Actions. To accept an intentional change, update the baseline file in
the same PR (and explain why in the PR description).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GATED = ("recall@5", "mrr")


def compare(current: dict, baseline: dict, max_drop: float) -> tuple[bool, str]:
    rows, ok = ["| Metric | Baseline | This PR | Change |", "|---|---|---|---|"], True
    for key in GATED:
        b, c = baseline.get(key), current.get(key)
        if b is None or c is None:
            rows.append(f"| {key} | {b} | {c} | n/a |")
            continue
        drop = b - c
        status = "❌ regression" if drop > max_drop else "✅"
        ok &= drop <= max_drop
        rows.append(f"| {key} | {b:.3f} | {c:.3f} | {c - b:+.3f} {status} |")
    return ok, "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--max-drop", type=float, default=0.02)
    args = parser.parse_args()

    current = json.loads(args.current.read_text(encoding="utf-8"))["summary"]
    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}; nothing to compare. Commit one to enable the gate.")
        return
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = baseline.get("summary", baseline)
    ok, md = compare(current, baseline, args.max_drop)
    report = f"### Retrieval eval (dev split)\n\n{md}\n"
    print(report)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(summary_file).write_text(report, encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
