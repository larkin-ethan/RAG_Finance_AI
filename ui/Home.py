"""FinRAG home. Run: uv run streamlit run ui/Home.py"""

import streamlit as st
from api import companies, sidebar_status

st.set_page_config(page_title="SEC Filings Q&A", page_icon="📈", layout="wide")
sidebar_status()

st.title("📈 SEC Filings Q&A")
st.write(
    "Ask questions about 10-Ks and 10-Qs and get accurate answers with **cited source "
    "passages**. When the filings don't support an answer, it says *I don't know*."
)

cols = st.columns(4)
cards = [
    ("📥", "Load Filings", "Add companies from SEC EDGAR", "pages/2_Load_Filings.py"),
    ("💬", "Ask Questions", "Answers with source citations", "pages/1_Chat.py"),
    ("🗂️", "Explore Filings", "Browse by company, year, or type", "pages/3_Browse.py"),
    ("📊", "Eval Results", "Accuracy, faithfulness, latency", "pages/5_Eval_Results.py"),
]
for col, (icon, title, desc, page) in zip(cols, cards):
    with col.container(border=True):
        st.markdown(f"### {icon}\n**{title}**")
        st.caption(desc)
        if page:
            st.page_link(page, label="Open →")

left, right = st.columns(2)
with left.container(border=True):
    st.subheader("Quick start")
    st.markdown(
        "1. **Load filings**: add a ticker and the app pulls its filings from EDGAR\n"
        "2. **Wait for processing**: parsing, chunking, and embedding (progress shown)\n"
        "3. **Ask**: questions are answered only from the filings, with citations"
    )
with right.container(border=True):
    st.subheader("Loaded companies")
    rows = companies()
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("None yet. Start with **Load Filings**.")
