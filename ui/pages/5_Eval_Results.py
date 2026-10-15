"""Eval dashboard: summary metrics for saved eval runs, baseline vs. others."""

import httpx
import streamlit as st
from api import get, sidebar_status

st.set_page_config(page_title="Eval Results · SEC Filings Q&A", page_icon="📊", layout="wide")
sidebar_status()
st.title("Evaluation Results")

try:
    runs = get("/evals")
except httpx.HTTPError as exc:
    st.error(f"API error: {exc}")
    st.stop()
if not runs:
    st.info("No eval runs yet. Run `python -m finrag.evals.run --name baseline` first.")
    st.stop()

PCT = ["recall@1", "recall@5", "recall@10", "answer_correctness", "faithfulness",
       "grounded_rate", "abstain_when_unanswerable", "false_abstention_rate"]  # fmt: skip
labels = [f"{r['summary']['config']['name']} · {r['file']}" for r in runs]

picked = st.multiselect("Runs to compare", labels, default=labels[: min(2, len(labels))])
chosen = [r for r, label in zip(runs, labels) if label in picked]
if not chosen:
    st.stop()

latest = chosen[-1]["summary"]
first = chosen[0]["summary"]
cards = [("recall@5", "Recall@5"), ("answer_correctness", "Answer correctness"),
         ("faithfulness", "Faithfulness"), ("abstain_when_unanswerable", "Abstains when it should")]  # fmt: skip
for col, (key, label) in zip(st.columns(len(cards)), cards):
    value, base = latest.get(key), first.get(key)
    delta = (
        f"{(value - base) * 100:+.1f} pts"
        if value is not None and base is not None and len(chosen) > 1
        else None
    )
    col.metric(label, f"{value:.1%}" if value is not None else "n/a", delta)

st.subheader("Comparison")
rows = []
for key in PCT + ["mrr", "latency_s_p50", "latency_s_p95", "cost_per_query_usd"]:
    row = {"metric": key}
    for r in chosen:
        v = r["summary"].get(key)
        row[r["summary"]["config"]["name"]] = (
            f"{v:.1%}" if key in PCT and v is not None else (f"{v:.3f}" if v is not None else "n/a")
        )
    rows.append(row)
st.dataframe(rows, hide_index=True, use_container_width=True)

st.subheader("Correctness by question type")
by_type = {
    r["summary"]["config"]["name"]: r["summary"].get("correctness_by_type", {}) for r in chosen
}
types = sorted({t for v in by_type.values() for t in v})
if types:
    st.dataframe(
        [{"type": t} | {n: f"{v.get(t, 0):.1%}" for n, v in by_type.items()} for t in types],
        hide_index=True, use_container_width=True,
    )  # fmt: skip
with st.expander("Run configs"):
    st.json({r["file"]: r["summary"]["config"] for r in chosen})
