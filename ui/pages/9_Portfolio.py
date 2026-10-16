"""Portfolio: your tax lots, tax profile, CSV import, and a summary of each holding."""

from datetime import date

import httpx
import streamlit as st
from api import delete, get, get_or_stop, post, put, require_ratings, sidebar_status

st.set_page_config(page_title="Portfolio · FinRAG", page_icon="💼", layout="wide")
sidebar_status()
require_ratings()
st.title("Portfolio")

tab_lots, tab_import, tab_tax, tab_report = st.tabs(
    ["Lots", "Import CSV", "Tax profile", "Summary"]
)

with tab_lots:
    lots = get_or_stop("/positions")
    if lots:
        st.dataframe(lots, hide_index=True, use_container_width=True,
                     column_order=["id", "ticker", "shares", "cost_per_share", "purchase_date", "account"])  # fmt: skip
        with st.form("delete"):
            pid = st.selectbox("Delete lot", [lot["id"] for lot in lots],
                               format_func=lambda i: next(f"#{x['id']} {x['ticker']} ({x['shares']} sh, {x['purchase_date']})"
                                                          for x in lots if x["id"] == i))  # fmt: skip
            if st.form_submit_button("Delete"):
                delete(f"/positions/{pid}")
                st.rerun()
    else:
        st.caption("No lots yet. Add one below or import a CSV from your broker.")
    with st.form("add", clear_on_submit=True):
        st.markdown("**Add a lot**")
        a, b, c, d, e = st.columns(5)
        t = a.text_input("Ticker")
        sh = b.number_input("Shares", min_value=0.0, step=1.0)
        cost = c.number_input("Cost per share ($)", min_value=0.0, step=0.01)
        bought = d.date_input("Purchase date", max_value=date.today())
        acct = e.text_input("Account (optional)")
        if st.form_submit_button("Add", type="primary"):
            try:
                post("/positions", {"ticker": t, "shares": sh, "cost_per_share": cost,
                                    "purchase_date": bought.isoformat(), "account": acct})  # fmt: skip
                st.rerun()
            except httpx.HTTPStatusError as exc:
                st.error(exc.response.json().get("detail"))

with tab_import:
    st.caption(
        "Export positions/lots from your broker as CSV (remove account numbers first). Max 1 MB."
    )
    up = st.file_uploader("CSV file", type=["csv"])
    if up:
        text = up.getvalue().decode("utf-8-sig", errors="replace")
        header = text.splitlines()[0].split(",") if text else []
        saved = get_or_stop("/csv-mappings")
        preset = st.selectbox("Saved mapping", ["(none)", *saved])
        base = saved.get(preset, {}) if preset != "(none)" else {}
        cols = ["(not in file)", *header]
        fields = {"ticker": "Ticker / symbol", "shares": "Shares / quantity", "cost_per_share": "Cost per share",
                  "total_cost": "Total cost (if no per-share cost)", "purchase_date": "Purchase date",
                  "account": "Account (optional)"}  # fmt: skip
        mapping = {}
        grid = st.columns(3)
        for i, (field, label) in enumerate(fields.items()):
            default = cols.index(base[field]) if base.get(field) in cols else 0
            choice = grid[i % 3].selectbox(label, cols, index=default, key=f"map_{field}")
            if choice != "(not in file)":
                mapping[field] = choice
        if "cost_per_share" in mapping:
            mapping.pop("total_cost", None)
        name = st.text_input("Save this mapping as (optional)", placeholder="e.g. Fidelity")
        if st.button("Preview"):
            st.session_state["import_preview"] = post(
                "/positions/import", {"csv_text": text, "mapping": mapping, "dry_run": True}
            )
        if pv := st.session_state.get("import_preview"):
            st.dataframe(pv["preview"], hide_index=True, use_container_width=True)
            for err in pv["errors"]:
                st.error(err)
            if (
                pv["preview"]
                and not pv["errors"]
                and st.button(f"Import {len(pv['preview'])} lots", type="primary")
            ):
                r = post("/positions/import", {"csv_text": text, "mapping": mapping, "dry_run": False,
                                               "save_mapping_as": name or None})  # fmt: skip
                st.success(f"Imported {r['imported']} lots.")
                st.session_state.pop("import_preview")

with tab_tax:
    st.caption(
        "Used only for federal tax estimates on this machine. Stored encrypted; never sent to any LLM."
    )
    current = get_or_stop("/tax-profile")
    statuses = {
        "single": "Single",
        "married_joint": "Married filing jointly",
        "head_of_household": "Head of household",
    }
    with st.form("tax"):
        fs = st.selectbox("Filing status", list(statuses), format_func=statuses.get,
                          index=list(statuses).index(current.get("filing_status", "single")))  # fmt: skip
        ti = st.number_input("Estimated taxable income this year, before any sale ($)", min_value=0.0, step=1000.0,
                             value=float(current.get("taxable_income", 0)))  # fmt: skip
        magi = st.number_input("Estimated MAGI ($, 0 = estimate from taxable income)", min_value=0.0, step=1000.0,
                               value=float(current.get("magi") or 0))  # fmt: skip
        if st.form_submit_button("Save", type="primary"):
            put("/tax-profile", {"filing_status": fs, "taxable_income": ti, "magi": magi or None})
            st.success("Saved.")

with tab_report:
    if st.button("Build summary"):
        with st.spinner("Checking each holding…"):
            for row in get("/portfolio/report"):
                if "error" in row:
                    st.error(f"{row['ticker']}: {row['error']}")
                    continue
                with st.container(border=True):
                    st.markdown(
                        f"**{row['ticker']}**"
                        + (f" · ${row['price']:,.2f}" if row["price"] else "")
                    )
                    st.markdown(" · ".join(f"{g['label']}: **{g['grade']}**" for g in row["goals"]))
                    gain = sum(v["gain"] or 0 for v in row["lots"])
                    tax = sum(v["tax_now"] or 0 for v in row["lots"])
                    st.caption(
                        f"Unrealized gain ${gain:,.0f} · est. federal tax if all sold today ${tax:,.0f}"
                    )
