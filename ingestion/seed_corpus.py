import json
import os
from pathlib import Path

# Ensure we're working from the project root
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"

# Seed corpus with rich metadata matching the plan specification
corpus = [
    {
        "id": "1",
        "title": "Causal Inference in Statistics: A Primer",
        "authors": ["Judea Pearl", "Madelyn Glymour", "Nicholas P. Jewell"],
        "abstract": "The back-door criterion provides a systematic approach for identifying causal effects from observational data. This method relies on blocking all confounding paths between treatment and outcome variables through proper covariate adjustment.",
        "venue": "Cambridge University Press",
        "year": 2016,
        "url": "https://www.cambridge.org/core/books/causal-inference-in-statistics/",
        "tags": ["theory", "DAG", "back-door"]
    },
    {
        "id": "2", 
        "title": "Identification and Estimation of Local Average Treatment Effects",
        "authors": ["Guido W. Imbens", "Joshua D. Angrist"],
        "abstract": "The front-door criterion enables causal identification when unobserved confounders are present but mediating variables can be observed. This approach is particularly useful when direct confounder control is impossible.",
        "venue": "Econometrica",
        "year": 1994,
        "url": "https://www.jstor.org/stable/2951620",
        "tags": ["theory", "front-door", "identification"]
    },
    {
        "id": "3",
        "title": "Instrumental Variables: What Are They and What Do They Do?",
        "authors": ["Joshua D. Angrist", "Alan B. Krueger"],
        "abstract": "Instrumental variables provide a method for estimating causal effects when randomized experiments are not feasible and confounding variables exist. The instrument must satisfy relevance and exclusion restriction assumptions.",
        "venue": "Journal of Economic Literature", 
        "year": 2001,
        "url": "https://www.aeaweb.org/articles?id=10.1257/jel.39.2.462",
        "tags": ["method", "IV", "econ"]
    },
    {
        "id": "4",
        "title": "Uplift Modeling for Direct Marketing",
        "authors": ["Piotr Rzepakowski", "Szymon Jaroszewicz"],
        "abstract": "Causal inference techniques are applied to measure heterogeneous treatment effects in advertising campaigns. Uplift modeling identifies customers who will respond positively to marketing interventions.",
        "venue": "KDD",
        "year": 2008, 
        "url": "https://dl.acm.org/doi/10.1145/1401890.1401936",
        "tags": ["applications", "ads", "CATE"]
    },
    {
        "id": "5",
        "title": "Double/Debiased Machine Learning for Treatment and Structural Parameters",
        "authors": ["Victor Chernozhukov", "Denis Chetverikov", "Mert Demirer", "Esther Duflo", "Christian Hansen", "Whitney Newey", "James Robins"],
        "abstract": "Double machine learning (DML) provides a framework for estimating causal parameters in high-dimensional settings. The method uses cross-fitting and Neyman orthogonality to achieve rate-optimal estimation.",
        "venue": "Econometrics Journal",
        "year": 2018,
        "url": "https://academic.oup.com/ectj/article/21/1/C1/5056401",
        "tags": ["method", "DML", "econ"]
    },
    {
        "id": "6",
        "title": "Causal Inference for Policy Evaluation and Optimal Policy Learning",
        "authors": ["Nathan Kallus", "Angela Zhou"],
        "abstract": "This work presents methods for evaluating treatment assignment policies and learning optimal policies from observational data. Applications include personalized medicine and targeted interventions.",
        "venue": "NeurIPS",
        "year": 2019,
        "url": "https://proceedings.neurips.cc/paper/2019",
        "tags": ["applications", "health", "policy"]
    },
    {
        "id": "7",
        "title": "Metalearners for Estimating Heterogeneous Treatment Effects",
        "authors": ["Sören R. Künzel", "Jasjeet S. Sekhon", "Peter J. Bickel", "Bin Yu"],
        "abstract": "Meta-learning approaches including T-learner, S-learner, and X-learner provide flexible frameworks for estimating conditional average treatment effects (CATE) using machine learning methods.",
        "venue": "PNAS",
        "year": 2019,
        "url": "https://www.pnas.org/doi/10.1073/pnas.1804597116", 
        "tags": ["method", "CATE", "health"]
    },
    {
        "id": "8",
        "title": "DoWhy: An End-to-End Library for Causal Inference",
        "authors": ["Amit Sharma", "Emre Kiciman"],
        "abstract": "DoWhy provides a unified framework for causal inference that emphasizes explicit assumptions, multiple estimation methods, and robustness testing. The library implements identification, estimation, and validation steps.",
        "venue": "arXiv",
        "year": 2020,
        "url": "https://arxiv.org/abs/2011.04216",
        "tags": ["method", "software", "DAG"]
    }
]

# Create data directory in project root
data_dir.mkdir(exist_ok=True)

# Write corpus to JSONL file
corpus_file = data_dir / "corpus.jsonl"
with open(corpus_file, "w") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")

print(f"Seed corpus with {len(corpus)} documents written to {corpus_file}")
print("Documents cover theory, methods, and applications across econ, health, and ads domains.")
