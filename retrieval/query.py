import pickle

with open("artifacts/bm25.pkl", "rb") as f:
    store = pickle.load(f)

bm25, docs = store["bm25"], store["docs"]

def search(query, k=3):
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])[:k]
    return ranked

if __name__ == "__main__":
    print(search("instrumental variables"))
