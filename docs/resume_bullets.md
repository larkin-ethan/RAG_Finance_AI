# Resume bullets (draft): fill in the bracketed numbers from the final eval

Pick 3–4. Lead with the outcome, then how. Every number must come from
`python -m finrag.evals.report evals/results/baseline-test.json evals/results/final-test.json`.

- Built a retrieval-augmented Q&A system over 120 SEC 10-K/10-Q filings (8 companies) that
  cites verbatim source passages and abstains when unsupported; answer correctness
  [X%] vs [Y%] for a naive-RAG baseline on a held-out test set.
- Designed a 94-question evaluation suite with gold answers from SEC XBRL financial data
  (independent of the pipeline), including multi-filing, cross-company, and unanswerable
  questions; measured recall@k, LLM-judged faithfulness, and abstention accuracy.
- Improved recall@5 from [A%] to [B%] with section-aware chunking, hybrid vector +
  keyword retrieval (reciprocal rank fusion), question-derived metadata filters, and
  cross-encoder reranking, each change measured separately on a dev split.
- Eliminated unsupported numbers in answers ([Z%] → [W%]) with a deterministic,
  unit-aware citation verifier that checks every quote and figure against its source.
- Fixed 11 real-world data issues found in production filings (split-span numbers,
  misaligned table spans, fiscal-calendar mismatches, EDGAR pagination), cutting
  corrupted numbers in parsed text from 259 to 0.
- Shipped on Google Cloud Run + Cloud SQL (pgvector) with a daily ingestion job, rate
  limiting, and CI that blocks PRs if retrieval recall drops more than 2 points.

Interview talking points: why gold comes from XBRL, why strict "I don't know", why RRF
needs no score calibration, and the NVIDIA fiscal-year filter trap.
