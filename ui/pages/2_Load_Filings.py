"""Load Filings: pull a company's 10-K/10-Q filings from EDGAR and index them."""

import time
from datetime import date, timedelta

import httpx
import streamlit as st
from api import companies, get, post, sidebar_status

st.set_page_config(page_title="Load Filings · SEC Filings Q&A", page_icon="📥", layout="wide")
sidebar_status()
st.title("Load SEC Filings")

with st.form("ingest"):
    ticker = st.text_input("Ticker", placeholder="e.g. AAPL").strip().upper()
    since = st.date_input("Filings since", date.today() - timedelta(days=3 * 365))
    submitted = st.form_submit_button("Start download & processing", type="primary")

STAGES = {"queued": 0.0, "listing": 0.05, "downloading": 0.1, "parsing": 0.4,
          "embedding": 0.55, "done": 1.0}  # fmt: skip

if submitted and ticker:
    try:
        job_id = post("/ingest", {"ticker": ticker, "since": since.isoformat()})["job_id"]
    except httpx.HTTPError as exc:
        st.error(f"Could not start: {exc}")
        st.stop()
    bar, status = st.progress(0.0), st.empty()
    while True:
        job = get(f"/ingest/{job_id}")
        stage, done, total = job["stage"], job["done"], job["total"]
        if stage == "error":
            bar.empty()
            st.error(f"Failed: {job['message']}")
            break
        base = STAGES.get(stage, 0.0)
        nxt = min((v for v in STAGES.values() if v > base), default=1.0)
        frac = base + (nxt - base) * (done / total if total else 0)
        bar.progress(min(frac, 1.0))
        status.caption(f"**{stage.title()}** {done}/{total} · {job['message']}")
        if stage == "done":
            st.success(f"Loaded {job['message']} ({done:,} passages indexed)")
            companies.clear()
            break
        time.sleep(1)

st.subheader("Loaded companies")
rows = companies()
if rows:
    st.dataframe(rows, hide_index=True, use_container_width=True)
else:
    st.caption("None yet.")
