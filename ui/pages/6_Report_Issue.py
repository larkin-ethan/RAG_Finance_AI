"""Report a bad answer that wasn't from the chat (e.g. noticed later)."""

import httpx
import streamlit as st
from api import post, sidebar_status

st.set_page_config(page_title="Report an Issue · SEC Filings Q&A", page_icon="🚩", layout="wide")
sidebar_status()
st.title("Report an Issue")
st.info("Found an incorrect or unhelpful answer? It's logged and reviewed to improve the system.")

ISSUES = {
    "incorrect_answer": "Incorrect answer",
    "wrong_or_missing_source": "Wrong or missing source",
    "made_up_number": "Made-up number",
    "should_have_said_idk": 'Should have said "I don\'t know"',
    "unhelpful": "Unhelpful",
    "other": "Other",
}
with st.form("report", clear_on_submit=True):
    question = st.text_input("Question", placeholder="What was Apple's revenue in 2024?")
    answer = st.text_area("The answer you got (optional)")
    issue = st.selectbox("Issue type", list(ISSUES), format_func=ISSUES.get)
    details = st.text_area(
        "What was wrong? (optional)", placeholder="e.g. wrong number, missing source"
    )
    if st.form_submit_button("Submit feedback", type="primary"):
        if not question.strip():
            st.error("Please enter the question.")
        else:
            try:
                post("/feedback", {"rating": -1, "question": question, "answer": answer or None,
                                   "issue_type": issue, "details": details or None})  # fmt: skip
                st.success("Thanks! Your report was logged.")
            except httpx.HTTPError as exc:
                st.error(f"Couldn't send: {exc}")
