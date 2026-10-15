"""Chat: ask questions, see cited answers and their source passages."""

import hashlib

import httpx
import streamlit as st
from api import companies, post, sidebar_status

st.set_page_config(page_title="Chat · SEC Filings Q&A", page_icon="💬", layout="wide")
sidebar_status()

# ---- filters (metadata filters narrow retrieval before vector search) ----
with st.sidebar:
    st.subheader("Filters")
    tickers = st.multiselect("Companies", [c["ticker"] for c in companies()])
    years = st.multiselect("Fiscal years", list(range(2026, 2019, -1)))
    forms = st.multiselect("Filing types", ["10-K", "10-Q"])
    k = st.slider("Passages to retrieve", 3, 12, 6)

head, clear = st.columns([5, 1])
head.title("Ask a Question")
if clear.button("Clear chat", use_container_width=True):
    st.session_state.messages = []

st.session_state.setdefault("messages", [])


ISSUES = {
    "incorrect_answer": "Incorrect answer",
    "wrong_or_missing_source": "Wrong or missing source",
    "made_up_number": "Made-up number",
    "should_have_said_idk": 'Should have said "I don\'t know"',
    "unhelpful": "Unhelpful",
    "other": "Other",
}


def send_feedback(ans: dict, rating: int, issue: str | None = None, details: str = "") -> None:
    try:
        post("/feedback", {
            "rating": rating, "question": ans["question"], "answer": ans["text"],
            "issue_type": issue, "details": details or None, "model": ans.get("model"),
            "citations": [{"number": c["number"], "quote": c["quote"], "verified": c["verified"],
                           "chunk_id": (c["hit"] or {}).get("chunk_id")} for c in ans["citations"]],
        })  # fmt: skip
        st.session_state[f"fb_done_{id_of(ans)}"] = True
    except httpx.HTTPError as exc:
        st.error(f"Couldn't send feedback: {exc}")


def id_of(ans: dict) -> str:
    """Stable widget key for an answer (hash() is randomized per process)."""
    return hashlib.sha256(f"{ans['question']}|{ans['text']}".encode()).hexdigest()[:12]


def feedback_controls(ans: dict) -> None:
    key = id_of(ans)
    if st.session_state.get(f"fb_done_{key}"):
        st.caption("Thanks for the feedback!")
        return
    up, down, _ = st.columns([1, 1, 10])
    if up.button("👍", key=f"up_{key}", help="Good answer"):
        send_feedback(ans, 1)
        st.rerun()
    if down.button("👎", key=f"down_{key}", help="Report a problem"):
        st.session_state[f"fb_open_{key}"] = True
    if st.session_state.get(f"fb_open_{key}"):
        with st.form(f"fb_form_{key}"):
            issue = st.selectbox("What went wrong?", list(ISSUES), format_func=ISSUES.get)
            details = st.text_area(
                "Details (optional)", placeholder="e.g. wrong number, missing source"
            )
            if st.form_submit_button("Submit feedback", type="primary"):
                send_feedback(ans, -1, issue, details)
                st.rerun()


def render_answer(ans: dict) -> None:
    st.markdown(ans["text"])
    if ans["answerable"] and not ans["grounded"]:
        st.warning(
            "Some parts of this answer couldn't be verified against the sources."
            + (
                f" Numbers not found verbatim: {', '.join(ans['unsupported_numbers'])}"
                if ans["unsupported_numbers"]
                else ""
            )
        )
    for c in ans["citations"]:
        hit = c["hit"] or {}
        period = f"FY{hit.get('fiscal_year')}" + (
            f" Q{hit['fiscal_quarter']}" if hit.get("fiscal_quarter") else ""
        )
        label = (
            f"{'✅' if c['verified'] else '⚠️'} [{c['number']}] {hit.get('company', '?')} "
            f"{hit.get('form', '')} {period} · Item {hit.get('section_key', '?')}"
        )
        with st.expander(label):
            st.markdown(f"> {c['quote']}")
            st.caption(hit.get("section_title", ""))
            links = []
            if hit.get("chunk_id"):
                accession = hit["chunk_id"].split(":")[0]
                links.append(f"[Open in document viewer](/Viewer?accession={accession}"
                             f"&chunk={hit['chunk_id']})")  # fmt: skip
            if hit.get("url"):
                links.append(f"[View on EDGAR ↗]({hit['url']})")
            st.markdown(" · ".join(links))
    meta = (
        f"{ans['model']} · {ans['input_tokens']:,} in / {ans['output_tokens']:,} out tokens · "
        f"retrieval {ans['retrieval_s']:.2f}s · generation {ans['generation_s']:.2f}s"
    )
    st.caption(meta + (" · cached" if ans.get("cached") else ""))
    feedback_controls(ans)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg["answer"])
        else:
            st.markdown(msg["text"])

if question := st.chat_input("Ask about a company's filings…"):
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Searching filings…"):
        try:
            ans = post(
                "/ask",
                {
                    "question": question,
                    "tickers": tickers,
                    "fiscal_years": years,
                    "forms": forms,
                    "k": k,
                },
            )
        except httpx.HTTPError as exc:
            st.error(f"API error: {exc}")
        else:
            render_answer(ans)
            st.session_state.messages.append({"role": "assistant", "answer": ans})
