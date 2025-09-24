import json, os

corpus = [
    {"id": "1", "title": "Back-Door Adjustment", "text": "The back-door criterion is a method for identifying causal effects."},
    {"id": "2", "title": "Front-Door Adjustment", "text": "The front-door criterion uses mediators to identify causal effects."},
    {"id": "3", "title": "Instrumental Variables", "text": "IV methods help identify causal effects when confounding is present."},
    {"id": "4", "title": "Uplift Modeling in Ads", "text": "Causal inference is applied to measure heterogeneous treatment effects in advertising."}
]

os.makedirs("data", exist_ok=True)
with open("data/corpus.jsonl", "w") as f:
    for doc in corpus:
        f.write(json.dumps(doc) + "\n")
print("Seed corpus written to data/corpus.jsonl")
