import json
from rank_bm25 import BM25Okapi
import pickle, os

with open("data/corpus.jsonl") as f:
    docs = [json.loads(line) for line in f]

tokenized = [d["text"].lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "docs": docs}, f)

print("BM25 index built and saved to artifacts/bm25.pkl")
