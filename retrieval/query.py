from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    from router.router import route_query
except ModuleNotFoundError:
    # When running this file directly (python retrieval/query.py), Python adds
    # retrieval/ to sys.path, not the project root. Add the project root so the
    # package import works, then retry.
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from router.router import route_query
try:
    from retrieval.reranker import rerank as ce_rerank
except ModuleNotFoundError:
    # When running directly, add project root and retry
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from retrieval.reranker import rerank as ce_rerank


def _project_paths() -> Tuple[Path, Path]:
    base = Path(__file__).resolve().parents[1]
    artifacts = base / "artifacts"
    return base, artifacts


def _load_bm25_index(artifacts_dir: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    index_path = artifacts_dir / "bm25_index.pkl"
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found at: {index_path}")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["docs"]


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _infer_doc_domain(tags: List[str]) -> str:
    lowered = {t.lower() for t in tags}
    if any(t in lowered for t in ["econ", "economics"]):
        return "econ"
    if any(t in lowered for t in ["ads", "advertising", "marketing"]):
        return "ads"
    if any(t in lowered for t in ["health", "healthcare", "clinical"]):
        return "health"
    return "general"


def _compute_boosts(doc: Dict[str, Any], route: Dict[str, Any], min_year: int, max_year: int) -> Tuple[float, Dict[str, float]]:
    boosts: Dict[str, float] = {}

    # Lane boost
    doc_lane = "applications" if "applications" in [t.lower() for t in doc.get("tags", [])] else (
        "theory" if "theory" in [t.lower() for t in doc.get("tags", [])] else "general"
    )
    if route["lane"] in ("theory", "applications") and route["lane"] == doc_lane:
        boosts["lane_match"] = 0.10

    # Domain boost
    doc_domain = _infer_doc_domain(doc.get("tags", []))
    if route["domain"] in ("econ", "ads", "health") and route["domain"] == doc_domain:
        boosts["domain_match"] = 0.15

    # Method boost
    method_tag_map = {
        "IV": ["iv", "instrumental"],
        "DAG": ["dag", "back-door", "front-door"],
        "DML": ["dml", "double machine learning"],
        "CATE": ["cate", "uplift", "heterogeneous"],
    }
    doc_tags_lower = {t.lower() for t in doc.get("tags", [])}
    m = route.get("method", "unknown")
    if m in method_tag_map and any(x in doc_tags_lower for x in method_tag_map[m]):
        boosts["method_match"] = 0.20

    # Venue boost (light)
    preferred_venues = {
        "econ": {"econometrica", "qje", "aej", "journal of economic literature", "econometrics journal"},
        "ads": {"kdd", "www", "wsdm", "recsys"},
        "health": {"nejm", "jama", "bmj", "lancet"},
    }
    venue = str(doc.get("venue", "")).lower()
    dom = route.get("domain", "general")
    if dom in preferred_venues and any(v in venue for v in preferred_venues[dom]):
        boosts["venue_preference"] = 0.05

    # Recency prior
    year = int(doc.get("year", 0) or 0)
    if year and max_year > min_year:
        norm = (year - min_year) / float(max_year - min_year)
    else:
        norm = 0.0
    recency_weight = 0.12 if route.get("lane") == "applications" else 0.06
    boosts["recency"] = recency_weight * norm

    total = sum(boosts.values())
    return total, boosts


def search(query: str, k: int = 5, rerank: bool = False, rerank_top_n: int = 20) -> List[Dict[str, Any]]:
    base, artifacts = _project_paths()
    bm25, docs = _load_bm25_index(artifacts)

    # Route the query
    routed = route_query(query).to_dict()

    # Pre-compute year bounds
    years = [int(d.get("year", 0) or 0) for d in docs if d.get("year")]
    min_year = min(years) if years else 0
    max_year = max(years) if years else 0

    # BM25 scores
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    results: List[Dict[str, Any]] = []
    for doc, bm25_score in zip(docs, scores):
        boost_total, boost_components = _compute_boosts(doc, routed, min_year, max_year)
        final_score = float(bm25_score) * (1.0 + boost_total)
        result = {
            **doc,
            "bm25_score": float(bm25_score),
            "boost_total": round(boost_total, 4),
            "boosts": {k: round(v, 4) for k, v in boost_components.items()},
            "final_score": round(final_score, 6),
            "router": routed,
        }
        results.append(result)

    results.sort(key=lambda r: r["final_score"], reverse=True)

    # Optional reranking with cross-encoder on top-N
    if rerank and results:
        initial_top = results[: min(rerank_top_n, len(results))]
        reranked = ce_rerank(query, initial_top)
        # Keep only top-k after reranking
        results = reranked[:k]
    else:
        results = results[:k]
    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Hybrid BM25 search with router-aware boosts and optional reranker")
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Top-k results")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
    parser.add_argument("--rerank-top-n", type=int, default=20, help="How many from first stage to rerank")
    args = parser.parse_args()

    queries = args.query or [
        "front-door criterion",
        "exclusion restriction in ads",
        "double machine learning for CATE on CPS data",
    ]

    all_results = []
    for q in queries:
        topk = search(q, k=args.k, rerank=args.rerank, rerank_top_n=args.rerank_top_n)
        if args.json:
            all_results.append({"query": q, "results": topk})
        else:
            print(f"\n🔍 Query: {q}")
            print("=" * 60)
            for i, r in enumerate(topk, 1):
                print(f"[{i}] {r.get('title', 'Untitled')}")
                print(f"    Venue: {r.get('venue', 'Unknown')} ({r.get('year', 'Unknown')})")
                line = f"    Scores → BM25: {r['bm25_score']:.4f}  Boost: +{r['boost_total']:.3f}  Final: {r['final_score']:.4f}"
                if 'rerank_score' in r:
                    line += f"  Rerank: {r['rerank_score']:.4f}"
                print(line)
                print(f"    Boosts: {r['boosts']}")
                print(f"    URL: {r.get('url', '')}")
    if args.json:
        print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    # Ensure CWD is project root so relative paths work if someone runs from elsewhere
    base, artifacts = _project_paths()
    os.chdir(base)
    _cli()
