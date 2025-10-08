import streamlit as st
from retrieval.query import search
from router.router import route_query

st.set_page_config(page_title="Causal RAG Explorer", page_icon="🧭", layout="wide")
st.title("Causal Inference RAG Explorer")
st.caption("Router-aware retrieval with optional reranking. Day 7 UI v1.")

with st.sidebar:
    st.header("Filters & Options")
    lane_opt = st.selectbox("Lane", ["auto", "theory", "applications"])
    domain_opt = st.selectbox("Domain", ["auto", "general", "econ", "health", "ads"])
    year_min = st.number_input("Min year", value=1900, step=1)
    top_k = st.slider("Top-k", min_value=3, max_value=10, value=5)
    use_rerank = st.checkbox("Use reranker (cross-encoder)", value=False)

query = st.text_input("Ask a question about causal inference")

col1, col2 = st.columns([3, 1])
with col1:
    run = st.button("Search", type="primary")
with col2:
    synth = st.button("Ask LLM to synthesize (Day 8)")

if run and query:
    routed = route_query(query).to_dict()

    # Override router with sidebar selections if not auto
    if lane_opt != "auto":
        routed["lane"] = lane_opt
    if domain_opt != "auto":
        routed["domain"] = domain_opt

    st.subheader("Router")
    st.write({k: routed[k] for k in ["lane", "domain", "method", "confidence"]})

    # Perform search (rerank optional)
    results = search(query, k=int(top_k), rerank=use_rerank)

    st.subheader("Results")
    if not results:
        st.info("No results.")
    for i, r in enumerate(results, 1):
        with st.container():
            st.markdown(f"### [{i}] {r.get('title','Untitled')}")
            meta = f"{r.get('venue','Unknown')} • {r.get('year','Unknown')}"
            st.caption(meta)

            # Scores row
            score_cols = st.columns(4)
            score_cols[0].metric("BM25", f"{r['bm25_score']:.3f}")
            score_cols[1].metric("Boost", f"+{r['boost_total']:.3f}")
            score_cols[2].metric("Final", f"{r['final_score']:.3f}")
            if 'rerank_score' in r:
                score_cols[3].metric("Rerank", f"{r['rerank_score']:.3f}")
            else:
                score_cols[3].metric("Rerank", "—")

            # Router badges
            badges = f"Lane: `{routed['lane']}` • Domain: `{routed['domain']}` • Method: `{routed['method']}`"
            st.write(badges)

            # Snippet
            abstract = str(r.get('abstract', ''))
            st.write(abstract[:500] + ("…" if len(abstract) > 500 else ""))
            st.write(r.get('url', ''))

if synth:
    st.warning("Synthesis is coming on Day 8: will summarize retrieved docs with citations.")
