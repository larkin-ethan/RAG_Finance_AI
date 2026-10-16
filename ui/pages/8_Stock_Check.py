"""Stock check: how well does a stock fit my goals, and over my timeline?"""

from datetime import date, timedelta

import httpx
import streamlit as st
from api import get, require_ratings, sidebar_status

st.set_page_config(page_title="Stock Check · FinRAG", page_icon="🎯", layout="wide")
sidebar_status()
require_ratings()
st.title("Stock Check")

GOALS = {
    "dividend": "Dividend income",
    "growth": "Growth",
    "value": "Value",
    "stability": "Stability / low risk",
}
COLORS = {"Strong fit": "🟢", "Moderate fit": "🟡", "Weak fit": "🟠", "Poor fit": "🔴"}

c1, c2, c3 = st.columns([1, 2, 2])
ticker = c1.text_input("Ticker", "KO").strip().upper()
goals = c2.multiselect("Goals", list(GOALS), default=["dividend"], format_func=GOALS.get)
timeline = c3.radio("Timeline", ["None", "Holding period", "Sell date"], horizontal=True)
params: dict = {"goals": goals}
if timeline == "Holding period":
    params["horizon_days"] = 30 * c3.slider("Months", 3, 60, 24)
elif timeline == "Sell date":
    params["sell_date"] = c3.date_input("Planned sell date", date.today() + timedelta(days=365),
                                        min_value=date.today() + timedelta(days=1)).isoformat()  # fmt: skip

if not (st.button("Check", type="primary") and ticker and goals):
    st.stop()
with st.spinner(f"Checking {ticker}…"):
    try:
        r = get(f"/check/{ticker}", **params)
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.json().get("detail", str(exc)))
        st.stop()

st.subheader(f"{r['ticker']} · price ${r['price']:,.2f}" if r["price"] else r["ticker"])
st.caption(f"Financials through {r['period_end']} · as of {r['as_of']}")

cols = st.columns(len(r["goals"]))
for col, g in zip(cols, r["goals"]):
    score = f"{g['score']:.0f}/100" if g["score"] is not None else ""
    col.metric(g["label"], f"{COLORS.get(g['grade'], '⚪')} {g['grade']}", score, delta_color="off")
    for cap in g["caps"]:
        col.caption(f"⛔ Capped: {cap}")

if r.get("explanation"):
    with st.container(border=True):
        st.markdown(r["explanation"]["text"])
        st.caption("Written by the local LLM, every number verified" if r["explanation"]["source"] == "llm"
                   else "Template summary")  # fmt: skip

for g in r["goals"]:
    with st.expander(f"What drove the {g['label']} grade"):
        rows = [{"Factor": c["label"], "Value": c["value"], "Score": c["score"], "Weight": c["weight"],
                 "Compared with": c["basis"]} for c in g["components"]]  # fmt: skip
        st.dataframe(rows, hide_index=True, use_container_width=True)

if h := r.get("horizon"):
    st.subheader(f"Timeline: {h['days']} days (to {h['end']})")
    a, b, c = st.columns(3)
    a.metric(
        "Horizon fit",
        h["fit_grade"],
        f"{h['fit_score']:.0f}/100" if h["fit_score"] else "",
        delta_color="off",
    )
    if h["stock"]:
        b.metric("Historical chance of a loss", f"{h['stock']['chance_of_loss']:.0%}",
                 f"S&P 500: {h['market']['chance_of_loss']:.0%}" if h.get("market") else None, delta_color="off")  # fmt: skip
        c.metric("Typical range (10th to 90th pct)", f"{h['stock']['p10']:+.0%} to {h['stock']['p90']:+.0%}",
                 f"median {h['stock']['median']:+.0%}", delta_color="off")  # fmt: skip
    st.caption(h["fundamentals_note"])
    if h["expected_dividends"]:
        total = sum(amt for _, amt in h["expected_dividends"])
        st.caption(f"Expected dividends in this window (based on the last year): ${total:.2f}/share "
                   f"over {len(h['expected_dividends'])} payments")  # fmt: skip
    for note in h["notes"]:
        st.info(note)

if risk := r.get("risk"):
    st.subheader("What changed in the risk factors")
    st.caption(
        f"{risk['new_share']:.0%} of this year's risk-factor text is new; {risk['removed_share']:.0%} of last year's was removed."
    )
    for flag in risk["red_flags"]:
        st.error(f"New red flag: {flag}")
    for p in risk["new_paragraphs"][:3]:
        st.markdown(f"> {p[:600]}{'…' if len(p) > 600 else ''}")

if r["lots"]:
    st.subheader("Your lots")
    st.dataframe([{
        "Shares": v["lot"]["shares"], "Cost/share": v["lot"]["cost_per_share"], "Bought": v["lot"]["purchase_date"],
        "Value": v["market_value"], "Gain": v["gain"], "Long-term from": v["long_term_date"],
        "Est. tax if sold now": v["tax_now"], "Est. tax at later date": v["tax_later"],
        "Later date": v["later_date"], "Saved by waiting": v["savings_by_waiting"],
    } for v in r["lots"]], hide_index=True, use_container_width=True)  # fmt: skip
    for v in r["lots"]:
        for n in v["notes"]:
            st.caption(f"• {n}")

for note in r["notes"]:
    st.caption(f"ℹ️ {note}")
