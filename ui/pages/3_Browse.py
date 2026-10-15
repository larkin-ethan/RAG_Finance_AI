"""Browse filings by company, form type, and fiscal year."""

import httpx
import streamlit as st
from api import companies, get, sidebar_status

st.set_page_config(page_title="Browse · SEC Filings Q&A", page_icon="🗂️", layout="wide")
sidebar_status()
st.title("Browse Filings")

c1, c2, c3 = st.columns(3)
ticker = c1.selectbox("Company", ["All"] + [c["ticker"] for c in companies()])
form = c2.selectbox("Filing type", ["All", "10-K", "10-Q"])
year = c3.selectbox("Fiscal year", ["All"] + list(range(2026, 2019, -1)))

try:
    rows = get(
        "/filings",
        ticker=None if ticker == "All" else ticker,
        form=None if form == "All" else form,
        year=None if year == "All" else year,
    )
except httpx.HTTPError as exc:
    st.error(f"API error: {exc}")
    st.stop()

for r in rows:
    period = f"FY{r['fiscal_year']}" + (f" Q{r['fiscal_quarter']}" if r["fiscal_quarter"] else "")
    r["period"] = period
    r["open"] = f"/Viewer?accession={r['accession_number']}"

st.dataframe(
    rows,
    hide_index=True,
    use_container_width=True,
    column_order=[
        "company",
        "ticker",
        "form",
        "period",
        "period_end",
        "filing_date",
        "open",
        "url",
    ],
    column_config={
        "open": st.column_config.LinkColumn("Viewer", display_text="Open"),
        "url": st.column_config.LinkColumn("EDGAR", display_text="EDGAR ↗"),
        "period_end": "Period end",
        "filing_date": "Filed",
    },
)
st.caption(f"Showing {len(rows)} filings")
