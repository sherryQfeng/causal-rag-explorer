def classify_query(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["iv", "instrumental", "regression", "double ml", "back-door", "front-door"]):
        return "theory"
    elif any(w in q for w in ["ads", "marketing", "healthcare", "application", "real world"]):
        return "application"
    return "general"

if __name__ == "__main__":
    print(classify_query("applications of double ml in ads"))
