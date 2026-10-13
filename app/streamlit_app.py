"""Streamlit demo: ask a question about SEC filings and inspect the evidence behind the answer.

    streamlit run app/streamlit_app.py

It talks to the FastAPI service (see src/citesec/api/main.py), it does not load any model itself.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from citesec import ui

st.set_page_config(page_title="CiteSEC", page_icon="📑", layout="wide")
st.title("CiteSEC")
st.caption("Citation-grounded answers over SEC filings. Every claim points to a filing page, and "
           "the system declines when the evidence is weak.")

with st.sidebar:
    st.header("Settings")
    api = st.text_input("API address", ui.DEFAULT_API)
    health = ui.api_health(api)
    if health is None:
        st.error("API not reachable")
    elif health["llm_reachable"]:
        st.success(f"Ready: {health['index_chunks']:,} chunks, model {health['model']}")
    else:
        st.warning("API is up but the language model is not reachable")
    st.subheader("Optional filters")
    companies = st.text_input("Companies (comma separated)", "",
                              help="Leave empty to search every indexed company.")
    years = st.text_input("Fiscal years", "", help="For example 2022, 2023")
    top_k = st.slider("Evidence blocks given to the model", 1, 5, 5)
    st.subheader("Try an example")
    for i, ex in enumerate(ui.EXAMPLES):
        if st.button(ex[:60] + ("..." if len(ex) > 60 else ""), key=f"ex{i}"):
            st.session_state["question"] = ex

question = st.text_area("Question", key="question", height=90,
                        placeholder="e.g. What was Amcor's net accounts receivable in FY2020?")
if st.button("Ask", type="primary", disabled=len(question.strip()) < 3):
    with st.spinner("Searching filings and generating a cited answer..."):
        try:
            st.session_state["result"] = ui.api_query(
                api, ui.build_payload(question, companies, years, top_k))
            st.session_state["error"] = None
        except RuntimeError as exc:
            st.session_state["result"], st.session_state["error"] = None, str(exc)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

res = st.session_state.get("result")
if res:
    if res["abstained"]:
        st.warning("**Not enough evidence to answer.** "
                   + ui.ABSTAIN_TEXT.get(res["abstain_reason"], "No answer was produced."))
    else:
        st.subheader("Answer")
        st.markdown(ui.format_answer(res["answer"]))
        if res["invalid_citations_removed"]:
            st.caption(f"{res['invalid_citations_removed']} citation(s) removed because they did "
                       "not point at a retrieved passage.")

    cols = st.columns(5)
    cols[0].metric("Confidence", f"{res['confidence']:.2f}",
                   help="Top reranker score, 0 to 1. Below the calibrated threshold the system "
                        "declines without calling the model.")
    for col, (stage, secs) in zip(cols[1:], ui.latency_rows(res["latency_ms"]) +
                                  [("total", res["latency_ms"]["total"] / 1000)], strict=True):
        col.metric(stage.capitalize(), f"{secs:.2f} s")
    with st.expander("Latency breakdown"):
        df = pd.DataFrame(ui.latency_rows(res["latency_ms"]), columns=["stage", "seconds"])
        st.bar_chart(df, x="stage", y="seconds")

    if res["citations"]:
        st.subheader("Cited evidence")
        for c in res["citations"]:
            label = ui.citation_label(c["doc_name"], c["page"], c["section"])
            mark = {True: "supports the claim", False: "weak support", None: ""}[c["supported"]]
            with st.expander(f"{label}  ({mark}, score {c['score']:.2f})"):
                st.write(c["snippet"])
                if c["page"] is not None:
                    st.caption(f"PDF page {c['page']} of {c['doc_name']}")
    if res["evidence"]:
        with st.expander(f"Everything retrieved ({len(res['evidence'])} passages)"):
            for e in res["evidence"]:
                label = ui.citation_label(e["doc_name"], e["page"], e["section"])
                st.markdown(f"**{label}**  score {e['score']:.2f}"
                            + ("  :green-badge[cited]" if e["cited"] else ""))
                st.caption(e["snippet"])
