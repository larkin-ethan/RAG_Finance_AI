# Eval set

| File | What | How it's made |
|---|---|---|
| `questions/generated.jsonl` | numeric, multi-filing, comparison, unanswerable | `python -m finrag.evals.build` (gold numbers from SEC XBRL) |
| `questions/human.jsonl` | qualitative "text" questions | written by hand (you!) |
| `questions/assistant_drafted.jsonl` | 16 qualitative questions drafted by Claude | **review each one**: confirm the answer on EDGAR, reword if it echoes the filing, then set `"reviewed": true` (or delete it) |

Each question stores a **frozen `split`** (`dev` ≈70% / `test` ≈30%, stratified by
type). Tune only on `dev`. Run `test` once, at the end, for the final results table.
Never change a question's split after you've looked at results.

## Writing a hand-written question (target: ~25 more)

1. Pick a topic a real investor would ask about: risks, strategy, segments,
   guidance language, legal proceedings, capital returns, supply chain.
2. Find the evidence passage:
   ```bash
   uv run python -m finrag.evals.find "share repurchase" --ticker MSFT --form 10-K --year 2025
   ```
3. Add one JSON line to `questions/human.jsonl` (copy an existing line):
   - `id`: unique, e.g. `text-msft-buybacks-fy2025`
   - `type`: `"text"`
   - `question`: phrased like a user, not copied from the filing
   - `reference_answer`: 1–2 sentences, from the passage
   - `gold`: `[{"accession_number": "...", "ticker": "MSFT", "evidence": "<exact phrase>", "fiscal_year": 2025}]`.
     The evidence is a short exact phrase from the passage (case and whitespace don't matter).
     Retrieval counts a chunk as relevant if it's from that filing and contains it.
   - leave `split` out; it's assigned from a hash of the `id`
4. Re-run the finder with your evidence phrase to confirm it matches.

**Avoid leakage:** don't phrase questions with the filing's exact wording (that
makes retrieval artificially easy), and write questions *before* looking at what
the system retrieves for them.

## Automated audit of generated numbers

`python -m finrag.evals.audit --show 10` checks every XBRL gold value sits on a
table row whose label matches its metric (e.g. revenue on a sales row). Current
result: **74/74 on a matching row**. This complements the human spot-check below.

## Reviewing generated questions

Generated questions start `"reviewed": false`. Spot-check ~15: open the gold
filing on EDGAR, find the number, and confirm the reference answer. Then flip
`reviewed` to `true`. Report the number you checked and any errors found in
the README's eval section. That's part of a rigorous eval.
