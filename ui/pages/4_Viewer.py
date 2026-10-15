"""Document viewer: a filing's parsed sections, with a cited passage highlighted.

Open from Browse, or from a citation in Chat (?accession=...&chunk=...).
"""

import html

import httpx
import streamlit as st
from api import get, sidebar_status

st.set_page_config(page_title="Viewer · SEC Filings Q&A", page_icon="📄", layout="wide")
sidebar_status()

accession = st.query_params.get("accession")
focus = st.query_params.get("chunk")
if not accession:
    st.info("Open a filing from **Browse**, or from a citation in **Chat**.")
    st.stop()

try:
    doc = get(f"/filings/{accession}")
except httpx.HTTPError as exc:
    st.error(f"Couldn't load filing {accession}: {exc}")
    st.stop()

period = f"FY{doc['fiscal_year']}" + (f" Q{doc['fiscal_quarter']}" if doc["fiscal_quarter"] else "")
st.title(f"{doc['company']} {doc['form']} ({period})")
st.link_button("View original on EDGAR ↗", doc["url"])

tab_doc, tab_meta, tab_chunks = st.tabs(["Document", "Metadata", "Chunks"])

with tab_doc:
    sections = doc["sections"]
    focus_section = next(
        (i for i, s in enumerate(sections) if any(c["chunk_id"] == focus for c in s["chunks"])), 0
    )
    toc, body = st.columns([1, 3])
    labels = [f"Item {s['key']}: {s['title']}" if s["key"] != "preamble" else s["title"]
              for s in sections]  # fmt: skip
    choice = toc.radio("Table of contents", range(len(sections)), index=focus_section,
                       format_func=lambda i: labels[i][:60])  # fmt: skip
    if sections:
        section = sections[choice]
        body.subheader(labels[choice])
        for c in section["chunks"]:
            if c["chunk_id"] == focus:
                # Highlight the cited passage (escaped: filing text is untrusted).
                body.markdown(
                    "<div style='border-left:4px solid #2563EB;background:#1E293B;"
                    f"padding:0.75em 1em;white-space:pre-wrap'>{html.escape(c['text'])}</div>",
                    unsafe_allow_html=True,
                )
                body.caption("▲ cited passage")
            else:
                body.markdown(c["text"])

with tab_meta:
    st.json({k: v for k, v in doc.items() if k != "sections"})

with tab_chunks:
    total = sum(len(s["chunks"]) for s in doc["sections"])
    st.caption(
        f"{total} chunks (section-aware, ≤512 tokens). This is exactly what retrieval searches."
    )
    for s in doc["sections"]:
        for c in s["chunks"]:
            with st.expander(f"{c['chunk_id'].split(':')[-1]} · Item {s['key']}"):
                st.text(c["text"])
