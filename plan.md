Love it. Here’s a **14-day / 2-hours-a-day** plan that **uses a frontier LLM for synthesis** but keeps the **core value in your curated retrieval + real data demos**. You’ll finish with a clean Streamlit UI, traceable citations, and runnable notebooks.

# 14-Day “Causal RAG + Frontier LLM” Plan

### Week 1 — Retrieval Core + LLM Synthesis (specialized)

**Day 1 – Project boot**

* Repo + venv, install: `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `pydantic`, `streamlit`, `tqdm`, `pandas`, `numpy`.
* Folders: `ingestion/ indexing/ retrieval/ rerank/ ui/ eval/ data/{raw,processed}/ apps/`.

**Day 2 – Curate seed corpus**

* Pull 50–80 **abstracts/metadata** (arXiv: `stat.ME`, `econ.EM`, `cs.LG`; PyWhy/DoWhy/EconML docs).
* Normalize to JSONL: `{id,title,abstract,venue,year,url,tags:[method,domain]}`.

**Day 3 – BM25 baseline**

* Build BM25 over abstracts. CLI test: queries like “front-door criterion”, “exclusion restriction”.

**Day 4 – Embeddings index (hybrid)**

* Encode abstracts (`all-mpnet-base-v2`), FAISS index.
* **Hybrid score = BM25 + λ·cosine**; tune λ on 10 hand-picked queries.

**Day 5 – LLM query router (frontier assist)**

* Heuristic + LLM fallback: classify query → `{lane: theory|applications, domain: econ|health|ads, method: IV|DML|DAG|CATE}`.
* Use router to bias retrieval (e.g., venue/domain boosts, recency prior).

**Day 6 – Lightweight reranker**

* Cross-encoder (MiniLM/MS-MARCO) over top-20 to top-5.
* Keep scoring transparent: show BM25, embed, rerank scores in result object.

**Day 7 – Streamlit UI v1**

* Input → top-5 cards (title, snippet, URL, tags, scores).
* Filters: lane (theory/apps), domain, year ≥ slider.
* Button: **“Ask LLM to synthesize (with citations)”** → pass retrieved chunks to LLM.

### Week 2 — Real Apps + Evaluation + Polish

**Day 8 – LLM synthesis w/ guardrails**

* Prompt template that **quotes ≤50 words** and **lists URLs**. If uncertain, say what’s missing.
* Add “temperature” + “citation strictness” controls in UI.

**Day 9 – Real data app #1 (econ)**

* Notebook: Lalonde or IHDP with **DoWhy/EconML** (ATE/CATE); markdown links to retrieved papers (DML, back-door).

**Day 10 – Real data app #2 (ads/tech)**

* Synthetic ad-uplift dataset or public Criteo sample; show uplift / policy evaluation caveats; cite KDD/industry posts.

**Day 11 – Real data app #3 (healthcare)**

* IHDP or MIMIC-like public subset; show IV/back-door example; include a small DAG figure.

**Day 12 – Evaluation harness**

* `eval/qa.jsonl` (25 queries) with `must_cite` signals.
* Metrics: Recall\@k, MRR (retrieval), **Faithfulness** = (citations present ∧ quoted span from retrieved). Simple script prints scores.

**Day 13 – UI polish**

* Side panel: recency boost, venue weights (JASA/NeurIPS ↑), lane/domain chips.
* Expand card → full abstract; copy-to-BibTeX; download results JSON.
* “Open in notebook” links to your apps/ notebooks.

**Day 14 – Ship & reflect**

* README (screens, quick-start), short demo GIF.
* “What I learned” section (hybrid wins, router impact, LLM failure modes).
* Push + pin on GitHub.

---

## Why this beats “just ask GPT”

* **Curated, domain-anchored corpus** → higher precision on causal jargon.
* **Transparent citations** → every claim traces to title+URL+quoted span.
* **Steerable routing** → theory vs apps, venue/domain weighting, recency slider.
* **Real data notebooks** → literature → executable code, not just prose.
* **You learn the stack** → retrieval, hybrid fusion, reranking, eval, UI.

---

## Minimal tasks per day (so you actually finish)

* **90 min build** (code or notebook).
* **20 min test** (one query + screenshot).
* **10 min log** (what worked / next).

If you want, I can drop a **Streamlit UI stub + router/ranker skeleton** into a zip like we did before, so Day 1–3 are pure plug-and-play.
