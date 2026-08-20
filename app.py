"""Streamlit search UI over the year-partitioned FTS indexes.

Modelled on tim-pinecone/sec-dense-fts, with one structural difference that drives the
whole layout: that app keeps everything in `__default__` and treats year as a `$in`
metadata filter. Here year *is* the namespace, so the year control selects which
namespaces to fan out over rather than narrowing a filter — and because the FTS API
takes one namespace per request, every multi-year search is a client-side fan-out whose
merge strategy has to be chosen deliberately. See `finvec/search.py` for why BM25 and
cosine need different merges.
"""

from __future__ import annotations

import streamlit as st

from finvec import search as S
from finvec.config import SEC_INDEX, TRANSCRIPTS_INDEX
from finvec.fts import client

DATASETS = {"SEC 10-K": SEC_INDEX, "Earnings calls": TRANSCRIPTS_INDEX}
YEARS = [str(y) for y in range(2004, 2026)]

st.set_page_config(page_title="SEC & Earnings Search", layout="wide")


@st.cache_resource
def get_pinecone():
    return client()


@st.cache_resource
def get_index(index_name: str):
    return get_pinecone().preview.index(name=index_name)


@st.cache_data(show_spinner=False)
def cached_embed(text: str) -> list[float]:
    return S.embed_query(text)


with st.sidebar:
    st.header("Scope")
    dataset_label = st.radio("Corpus", list(DATASETS), horizontal=False)
    index_name = DATASETS[dataset_label]

    # Namespaces, not a filter. Each selected year is one additional search request.
    sel_years = st.multiselect(
        "Year namespaces", YEARS, default=["2024"],
        help="Each year is a separate namespace and a separate search request. "
             "Narrow this — read units scale with the number of namespaces queried.",
    )
    ticker = st.text_input("Ticker", placeholder="e.g. NVDA").strip() or None
    include_boilerplate = st.checkbox(
        "Include boilerplate", value=False,
        help="The source flags LSH-detected boilerplate. Excluded by default.",
    )
    top_k = st.slider("Results", 3, 50, 10)

    st.divider()
    st.caption(f"index: `{index_name}`")
    if len(sel_years) > 1:
        st.caption(f"fan-out: {len(sel_years)} concurrent searches")

st.title("SEC filings & earnings calls")
st.caption(
    "Full-text, semantic, and hybrid search over year-partitioned Pinecone FTS indexes."
)

mode_label = st.radio(
    "Search mode",
    ["Hybrid (semantic rank + must-contain)", "Full-text (BM25)", "Semantic (dense)"],
    horizontal=True,
)
MODES = {
    "Full-text (BM25)": "text",
    "Semantic (dense)": "dense",
    "Hybrid (semantic rank + must-contain)": "hybrid",
}
mode = MODES[mode_label]

merge_choice = "auto"
if len(sel_years) > 1:
    merge_choice = st.radio(
        "Merge across namespaces",
        ["auto", "score", "rrf", "rerank"],
        horizontal=True,
        help=(
            "auto: score for dense/hybrid, rank fusion for BM25. "
            "score is exact for cosine but biased for BM25, whose scores depend on "
            "per-namespace corpus statistics. rerank orders the union with a "
            "cross-encoder and is the only globally consistent option."
        ),
    )

st.divider()

if mode == "hybrid":
    col_a, col_b = st.columns(2)
    with col_a:
        query = st.text_input(
            "Semantic query (drives ranking)",
            placeholder="e.g. cloud infrastructure investment",
        )
    with col_b:
        must_contain = st.text_input(
            "Must-contain keywords (hard filter)",
            placeholder="e.g. Azure capital expenditure",
            help="Every token must be present. Applied as $match_all, so it constrains "
                 "results rather than influencing the score.",
        )
else:
    query = st.text_input(
        "Query",
        placeholder=(
            "e.g. revenue growth operating income" if mode == "text"
            else "e.g. risks related to supply chain disruption"
        ),
    )
    must_contain = None


def render(hits: list[S.Hit], strategy: str) -> None:
    if not hits:
        st.info("No results. Try widening the years, or unchecking 'exclude boilerplate'.")
        return

    st.caption(
        f"{len(hits)} result(s) across {len({h.namespace for h in hits})} namespace(s) "
        f"· merged by **{strategy}**"
    )
    if strategy == "rrf":
        st.warning(
            "BM25 scores are not comparable across namespaces — they depend on each "
            "namespace's own term statistics — so these are interleaved by rank, which "
            "gives every year equal representation. Pick **rerank** for a single global "
            "ordering.",
            icon="⚠️",
        )

    for hit in hits:
        f = hit.fields
        head = str(f.get("ticker", "?")).upper()
        when = f.get("call_date") or f.get("fiscal_year", "")
        quarter = f" · Q{int(f['quarter'])}" if f.get("quarter") else ""
        speaker = f" · {f['speaker']}" if f.get("speaker") else ""
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"**{head}** · {when}{quarter}{speaker}")
                tags = []
                if f.get("is_table"):
                    tags.append("table")
                if f.get("is_boilerplate"):
                    tags.append("boilerplate")
                if f.get("merged_from"):
                    tags.append(f"{int(f['merged_from'])} fragments")
                st.caption(
                    f"namespace `{hit.namespace}` · rank {hit.rank} in namespace"
                    + (" · " + " · ".join(tags) if tags else "")
                )
            with right:
                st.metric("score", f"{hit.ordering_score:.4f}",
                          label_visibility="collapsed")
            text = hit.text
            st.markdown(
                f"<small>{text[:400]}{'…' if len(text) > 400 else ''}</small>",
                unsafe_allow_html=True,
            )
            if len(text) > 400:
                with st.expander("Full chunk"):
                    st.write(text)


disabled = not query or not sel_years
if st.button("Search", type="primary", disabled=disabled):
    with st.spinner(
        f"Searching {len(sel_years)} namespace(s)…"
        if len(sel_years) > 1 else "Searching…"
    ):
        hits, strategy = S.search(
            get_pinecone(),
            get_index(index_name),
            query,
            sorted(sel_years),
            mode=mode,  # type: ignore[arg-type]
            top_k=top_k,
            must_contain=must_contain,
            ticker=ticker,
            include_boilerplate=include_boilerplate,
            merge=None if merge_choice == "auto" else merge_choice,  # type: ignore[arg-type]
        )
    render(hits, strategy)
elif not sel_years:
    st.info("Pick at least one year namespace in the sidebar.")
