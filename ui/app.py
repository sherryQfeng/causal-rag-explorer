import streamlit as st
from retrieval.query import search
from router.router import classify_query

st.title("Causal Inference RAG Explorer")

query = st.text_input("Ask a question about causal inference:")

if query:
    domain = classify_query(query)
    st.write(f"Query classified as: **{domain}**")
    results = search(query, k=3)
    for doc, score in results:
        st.markdown(f"### {doc['title']}")
        st.write(doc['text'])
        st.caption(f"Score: {score:.2f}")
