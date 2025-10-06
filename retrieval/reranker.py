from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

_MODEL_SINGLETON = None


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        from sentence_transformers import CrossEncoder
        _MODEL_SINGLETON = CrossEncoder(model_name)
    return _MODEL_SINGLETON


def build_pairs(query: str, docs: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for d in docs:
        title = str(d.get("title", "")).strip()
        abstract = str(d.get("abstract", "")).strip()
        text = f"{title}. {abstract}" if title else abstract
        pairs.append((query, text))
    return pairs


def rerank(query: str, candidates: List[Dict[str, Any]], batch_size: int = 32, model_name: str | None = None) -> List[Dict[str, Any]]:
    """Annotate candidates with 'rerank_score' using a cross-encoder and return them sorted by it."""
    if not candidates:
        return candidates

    model = get_cross_encoder(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = build_pairs(query, candidates)
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    for cand, s in zip(candidates, scores):
        cand["rerank_score"] = float(s)

    candidates.sort(key=lambda r: r["rerank_score"], reverse=True)
    return candidates


