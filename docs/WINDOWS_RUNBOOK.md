# Windows runbook: Phase A (load, eval, export)

Run these in **PowerShell** in order, on the Windows PC with the RTX GPU. Each step
says what "done" looks like. Expected total: about half a day, mostly waiting.

## 1. Install tools (once)

1. **Git:** https://git-scm.com/download/win
2. **Docker Desktop** with the WSL2 backend: https://www.docker.com/products/docker-desktop/
   (open it once and accept its terms).
3. **uv:**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Then open a **new** PowerShell window so `uv` is on PATH.
4. *(Optional, for the CI index release)* **GitHub CLI:** `winget install GitHub.cli`, then `gh auth login`.

## 2. Get the code

```powershell
git clone https://github.com/<you>/RAG_Finance_AI.git
cd RAG_Finance_AI
copy .env.example .env
notepad .env
```

In `.env`, set:
- `SEC_USER_AGENT="Your Name you@example.com"`
- `GEMINI_API_KEY=...` (from https://aistudio.google.com)
- leave `EMBED_PROVIDER` unset (defaults to `local`)

Never commit `.env`.

## 3. Install dependencies and check the GPU

```powershell
uv sync --extra dev
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
✅ Done when this prints `True` and your GPU's name. On Windows, uv installs the CUDA 12.8
build of PyTorch automatically (configured in `pyproject.toml`).

```powershell
uv run python -m finrag.tasks test
```
✅ All tests pass. The two database tests skip until step 4.

## 4. Start Postgres (pgvector)

```powershell
docker compose -f docker/docker-compose.yml up -d
docker ps
```
✅ A `docker-db-1` container shows `Up`. Run the tests again: the database tests should now run too.

## 5. Build the corpus (fetch → parse → chunk → embed → load)

```powershell
uv run python -m finrag.tasks pipeline
```
This downloads ~120 filings (about 10 minutes, mostly JPM), parses them, chunks them three
ways, and embeds and loads four strategies on the GPU. The first run downloads the Qwen
embedding model (about 1.2 GB).

✅ Done when the last step prints `Audited 74 gold values: 74 on a matching row`.

**Quick sanity check:**
```powershell
uv run python -m finrag.retrieve.search "What was Apple's total net sales in fiscal 2024?" --tickers AAPL --years 2024
```
The top hit should be Apple's net sales table (`Total net sales | $391,035`).

## 6. Run the evals

```powershell
uv run python -m finrag.tasks sweep
```
This is the retrieval-only comparison of 8 configurations. It's free, with no LLM calls.
**Note which configuration wins on recall@5.** If it isn't
`hybrid+filters+rerank`, edit the `final` line in `src/finrag/tasks.py` (or run
`finrag.evals.run` by hand with the winning flags).

```powershell
uv run python -m finrag.tasks evals
```
This is the full eval with Gemini answers and the judge, for baseline vs. final on the **dev**
split. If the free-tier daily quota runs out, re-run tomorrow: every LLM call is cached,
so it resumes where it stopped.

**Then, once only, on the held-out test split:**
```powershell
uv run python -m finrag.evals.run --name baseline --split test --strategy fixed-512-64
uv run python -m finrag.evals.run --name final --split test --method hybrid --filters auto --rerank
uv run python -m finrag.evals.report evals/results/baseline-test.json evals/results/final-test.json
```
Paste that table into the README.

## 7. Judge spot-check (~20 minutes)

Open `evals/results/final-dev.json`. For 20 questions with `"correct_method": "judge"`,
read the question, the answer, and `correct_reason`, and decide yourself whether the
answer is right. Record how often you agree with the judge (e.g. 18/20 = 90%) in the
README's eval section.

## 8. CI retrieval index

```powershell
uv run python -m finrag.evals.index export --strategy section-512
gh release create eval-index finrag-index.tsv.gz --title "Eval index" --notes "Retrieval index for CI"
copy evals\results\sweep\section-aware-dev.json evals\baseline_retrieval.json
git add evals/baseline_retrieval.json evals/results
git commit -m "Add eval results and CI retrieval baseline"
git push
```
✅ On the next PR, the "Retrieval eval" CI job runs instead of skipping.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Update the NVIDIA driver; run `uv sync --reinstall-package torch` |
| `connection refused` to Postgres | Start Docker Desktop, then step 4 again |
| `UnicodeEncodeError` | Shouldn't happen (output is forced to UTF-8); if it does, run `$env:PYTHONUTF8=1` and retry |
| `429` from Gemini | Free-tier quota. Wait (daily quota resets at midnight Pacific) and re-run; finished calls are cached |
| SEC `403` | `SEC_USER_AGENT` in `.env` must contain a real email |
