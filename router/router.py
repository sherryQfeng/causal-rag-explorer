from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RouterOutput:
    lane: str
    domain: str
    method: str
    confidence: float
    rationale: Dict[str, Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "lane": self.lane,
            "domain": self.domain,
            "method": self.method,
            "confidence": round(float(self.confidence), 3),
            "rationale": self.rationale,
        }


# Keyword inventories with light weights. Higher weight = more decisive.
LANE_THEORY_KEYWORDS: Dict[str, float] = {
    # Theory terms
    "back-door": 2.0,
    "front-door": 2.0,
    "d-separation": 2.0,
    "do-calculus": 2.0,
    "identifiability": 1.8,
    "theorem": 1.4,
    "assumption": 1.2,
    "criterion": 1.2,
    "proof": 1.2,
    "dag": 1.6,
    "causal graph": 1.4,
    "structural causal model": 1.6,
    "scm": 1.2,
    "instrumental variable": 1.6,
    "instrument": 1.0,
    "2sls": 1.2,
}

LANE_APPS_KEYWORDS: Dict[str, float] = {
    # Application/data/ops terms
    "ihdp": 1.6,
    "lalonde": 1.6,
    "mimic": 1.4,
    "criteo": 1.4,
    "campaign": 1.2,
    "ab test": 1.6,
    "a/b test": 1.6,
    "experiment": 1.2,
    "evaluation": 1.0,
    "case study": 1.0,
    "production": 1.0,
    "deployment": 1.0,
    "real world": 1.0,
    "healthcare": 1.2,
    "ads": 1.4,
    "marketing": 1.2,
    "economics": 1.0,
}

DOMAIN_KEYWORDS: Dict[str, Dict[str, float]] = {
    "econ": {
        "lalonde": 2.0,
        "cps": 1.6,
        "psid": 1.4,
        "nber": 1.4,
        "angrist": 1.6,
        "imbens": 1.6,
        "econml": 1.4,
        "aej": 1.2,
        "qje": 1.2,
        "econometrics": 1.2,
        "policy evaluation": 1.2,
    },
    "health": {
        "ihdp": 1.8,
        "mimic": 1.8,
        "rct": 1.6,
        "clinical": 1.4,
        "cohort": 1.4,
        "patient": 1.2,
        "nejm": 1.2,
        "icu": 1.2,
    },
    "ads": {
        "ads": 1.8,
        "advertising": 1.6,
        "marketing": 1.6,
        "campaign": 1.6,
        "ctr": 1.4,
        "cpa": 1.2,
        "uplift": 1.6,
        "criteo": 1.8,
        "ab test": 1.6,
        "a/b test": 1.6,
    },
}

METHOD_KEYWORDS: Dict[str, Dict[str, float]] = {
    "IV": {
        "instrumental variable": 2.0,
        "instrument": 1.4,
        "exclusion restriction": 2.0,
        "relevance": 1.2,
        "2sls": 1.6,
        "two-stage least squares": 1.6,
        "wald": 1.2,
        "late": 1.2,
    },
    "DAG": {
        "dag": 1.8,
        "causal graph": 1.6,
        "back-door": 2.0,
        "front-door": 2.0,
        "d-separation": 2.0,
        "do-calculus": 2.0,
        "structural causal model": 1.6,
        "scm": 1.2,
        "identifiability": 1.4,
    },
    "DML": {
        "double machine learning": 2.0,
        "orthogonal": 1.6,
        "nuisance": 1.4,
        "cross-fitting": 1.6,
        "partialling out": 1.4,
        "neyman": 1.2,
        "riesz": 1.2,
        "cher nozhukov": 1.2,  # typo-tolerant
        "cherno zhukov": 1.2,
        "chernozhukov": 1.2,
    },
    "CATE": {
        "cate": 1.8,
        "heterogeneous treatment": 1.8,
        "t-learner": 1.6,
        "s-learner": 1.6,
        "x-learner": 1.6,
        "meta-learner": 1.4,
        "uplift": 1.6,
        "treatment heterogeneity": 1.4,
        "policy learning": 1.2,
    },
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _score_keywords(text: str, keywords: Dict[str, float]) -> Tuple[float, List[str]]:
    """Return total weight and list of matched phrases."""
    score = 0.0
    hits: List[str] = []
    for phrase, weight in keywords.items():
        if phrase in text:
            score += weight
            hits.append(phrase)
    return score, hits


def _argmax(scores: Dict[str, Tuple[float, List[str]]], default_label: str, min_score: float = 0.9) -> Tuple[str, float, List[str], float]:
    """Pick top label with its score and matches; return runner-up score for confidence.

    If top score < min_score, fall back to default_label.
    """
    if not scores:
        return default_label, 0.0, [], 0.0
    ordered = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    top_label, (top_score, top_hits) = ordered[0]
    runner_up = ordered[1][1][0] if len(ordered) > 1 else 0.0
    if top_score < min_score:
        return default_label, 0.0, [], runner_up
    return top_label, top_score, top_hits, runner_up


def route_query(query_text: str) -> RouterOutput:
    """Heuristically classify a query into lane, domain, method with confidence and rationale."""
    q = _normalize(query_text)

    # Lane scoring
    lane_scores: Dict[str, Tuple[float, List[str]]] = {}
    theory_score, theory_hits = _score_keywords(q, LANE_THEORY_KEYWORDS)
    apps_score, apps_hits = _score_keywords(q, LANE_APPS_KEYWORDS)
    lane_scores["theory"] = (theory_score, theory_hits)
    lane_scores["applications"] = (apps_score, apps_hits)
    lane_label, lane_top, lane_hits, lane_second = _argmax(lane_scores, default_label="general", min_score=0.9)

    # Domain scoring
    domain_scores: Dict[str, Tuple[float, List[str]]] = {}
    for domain_label, kw in DOMAIN_KEYWORDS.items():
        score, hits = _score_keywords(q, kw)
        domain_scores[domain_label] = (score, hits)
    domain_label, domain_top, domain_hits, domain_second = _argmax(domain_scores, default_label="general", min_score=1.0)

    # Method scoring
    method_scores: Dict[str, Tuple[float, List[str]]] = {}
    for method_label, kw in METHOD_KEYWORDS.items():
        score, hits = _score_keywords(q, kw)
        method_scores[method_label] = (score, hits)
    method_label, method_top, method_hits, method_second = _argmax(method_scores, default_label="unknown", min_score=1.0)

    # Confidence as mean of per-axis normalized margins
    def margin_conf(top: float, second: float) -> float:
        if top <= 0.0:
            return 0.0
        # Normalized margin in [0,1]
        denom = top + second if (top + second) > 0 else top
        return max(0.0, min(1.0, (top - second) / denom))

    lane_conf = margin_conf(lane_top, lane_second)
    domain_conf = margin_conf(domain_top, domain_second)
    method_conf = margin_conf(method_top, method_second)
    overall_conf = (lane_conf + domain_conf + method_conf) / 3.0

    rationale = {
        "lane": {
            "label": lane_label,
            "score": round(lane_top, 3),
            "runner_up": round(lane_second, 3),
            "matches": lane_hits,
        },
        "domain": {
            "label": domain_label,
            "score": round(domain_top, 3),
            "runner_up": round(domain_second, 3),
            "matches": domain_hits,
        },
        "method": {
            "label": method_label,
            "score": round(method_top, 3),
            "runner_up": round(method_second, 3),
            "matches": method_hits,
        },
    }

    return RouterOutput(
        lane=lane_label,
        domain=domain_label,
        method=method_label,
        confidence=overall_conf,
        rationale=rationale,
    )


def classify_query(query: str) -> str:
    """Backward-compatible simple lane classifier used earlier in the project."""
    routed = route_query(query)
    return routed.lane


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Heuristic LLM query router")
    parser.add_argument("query", nargs="*", help="Query text to classify")
    args = parser.parse_args()
    queries = args.query or [
        "front-door criterion",
        "exclusion restriction in ads",
        "double machine learning for CATE on CPS data",
    ]
    results: List[Dict[str, object]] = []
    for q in queries:
        out = route_query(q)
        results.append({"query": q, **out.to_dict()})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _cli()
