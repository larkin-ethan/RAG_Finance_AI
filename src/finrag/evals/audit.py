"""CLI: sanity-check XBRL-generated gold answers against the parsed filings.

  python -m finrag.evals.audit [--show 10]

For every gold item, finds the table row containing the evidence number in the
gold filing and checks that the row's label matches the metric (e.g. a
"revenue" gold must sit on a sales/revenue row). This catches a number that
was found on an unrelated row that happens to share it. It complements, but
doesn't replace, a human spot-check against EDGAR (see evals/README.md).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict

from finrag.evals.dataset import load_questions
from finrag.evals.run import QUESTION_FILES
from finrag.generate.citations import normalize
from finrag.settings import get_settings

ROW_LABELS = {
    "revenue": r"revenue|net sales|sales",
    "net_income": r"net income|net earnings|earnings attributable",
    "eps_diluted": r"diluted",
    "operating_income": r"operating income|income from operations",
    "total_assets": r"total assets",
    "cash": r"cash and cash equivalents|cash and due",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0, help="print N random rows to eyeball")
    args = parser.parse_args()

    lines_by_filing: dict[str, list[str]] = defaultdict(list)
    path = get_settings().data_dir / "chunks" / "section-512.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            lines_by_filing[c["accession_number"]] += c["text"].split("\n")

    checked, problems, samples = 0, [], []
    for q in load_questions(*QUESTION_FILES):
        if q.source != "xbrl":
            continue
        metric = next((m for m in ROW_LABELS if f"-{m}-" in q.id), None)
        for g in q.gold:
            checked += 1
            ev = normalize(g.evidence)
            rows = [r for r in lines_by_filing[g.accession_number] if ev in normalize(r)]
            if not metric or not any(re.search(ROW_LABELS[metric], r, re.IGNORECASE) for r in rows):
                problems.append(f"{q.id}: {g.evidence} -> {rows[0][:120] if rows else 'not found'}")
            elif rows:
                samples.append(f"{q.question}\n   {g.evidence} on row: {rows[0][:120]}")

    for p in problems:
        print("?? " + p)
    for s in random.Random(0).sample(samples, min(args.show, len(samples))):
        print(s)
    print(f"\nAudited {checked} gold values: {checked - len(problems)} on a matching row, "
          f"{len(problems)} suspect")  # fmt: skip
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
