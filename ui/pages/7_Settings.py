"""Read-only view of the running configuration.

Deliberately read-only: in a public demo, letting anyone change models or
clear the database from the UI would be a security hole. Change settings via
environment variables / .env and redeploy.
"""

import httpx
import streamlit as st
from api import API_URL, get, sidebar_status

st.set_page_config(page_title="Settings · SEC Filings Q&A", page_icon="⚙️", layout="wide")
sidebar_status()
st.title("Settings")

try:
    cfg = get("/config")
except httpx.HTTPError as exc:
    st.error(f"API error: {exc}")
    st.stop()

left, right = st.columns(2)
with left.container(border=True):
    st.subheader("Models")
    st.text_input("LLM (answers)", cfg["llm_model"], disabled=True)
    st.text_input("LLM (eval judge)", cfg["judge_model"], disabled=True)
    st.text_input("Embeddings", f"{cfg['embed_model']} ({cfg['embed_provider']}, {cfg['embed_dim']}-dim)",
                  disabled=True)  # fmt: skip
with right.container(border=True):
    st.subheader("Retrieval & data")
    st.text_input("Chunking strategy", cfg["chunk_strategy"], disabled=True)
    st.text_input("Watchlist", cfg["watchlist"], disabled=True)
    st.text_input("API", API_URL, disabled=True)
st.caption("Settings are read-only here. Change them via environment variables and redeploy.")
