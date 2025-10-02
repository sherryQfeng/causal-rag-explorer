#!/usr/bin/env python3
"""
Day 4: Hybrid BM25 + Embeddings Implementation
Build hybrid search combining BM25 and semantic embeddings with FAISS.
"""

import json
import pickle
import os
import argparse
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Core libraries
from rank_bm25 import BM25Okapi
import faiss
from sentence_transformers import SentenceTransformer

def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Load corpus from JSONL file."""
    docs = []
    with open(corpus_path, 'r') as f:
        for line in f:
            docs.append(json.loads(line.strip()))
    return docs

def build_bm25_index(docs: List[Dict[str, Any]]) -> BM25Okapi:
    """Build BM25 index over document abstracts."""
    print(f"Building BM25 index over {len(docs)} documents...")
    
    # Tokenize abstracts
    tokenized_abstracts = []
    for doc in docs:
        abstract = doc.get('abstract', '')
        tokens = abstract.lower().split()
        tokenized_abstracts.append(tokens)
        
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_abstracts)
    return bm25

def build_embedding_index(docs: List[Dict[str, Any]], model_name: str = 'all-mpnet-base-v2') -> Tuple[np.ndarray, faiss.Index, SentenceTransformer]:
    """Build embedding index using sentence transformers and FAISS."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Encoding {len(docs)} document abstracts...")
    abstracts = [doc.get('abstract', '') for doc in docs]
    
    # Generate embeddings with progress bar
    embeddings = model.encode(abstracts, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    print(f"Building FAISS index (dimension: {embeddings.shape[1]})")
    # Use IndexFlatIP for cosine similarity (after L2 normalization)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    print(f"✅ FAISS index built with {index.ntotal} vectors")
    return embeddings, index, model

def hybrid_search(
    query: str, 
    docs: List[Dict[str, Any]], 
    bm25: BM25Okapi, 
    faiss_index: faiss.Index, 
    embedding_model: SentenceTransformer,
    lambda_param: float = 0.5,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Perform hybrid search combining BM25 and embeddings."""
    
    # BM25 search
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Embedding search
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # FAISS returns (distances, indices) - distances are cosine similarities
    cosine_scores, _ = faiss_index.search(query_embedding.astype('float32'), len(docs))
    cosine_scores = cosine_scores[0]  # Get first (and only) query result
    
    # Normalize scores to [0, 1] range for fair combination
    bm25_normalized = normalize_scores(bm25_scores)
    cosine_normalized = normalize_scores(cosine_scores)
    
    # Hybrid scoring: BM25 + λ * cosine
    hybrid_scores = bm25_normalized + lambda_param * cosine_normalized
    
    # Create results with all scores
    results = []
    for i, doc in enumerate(docs):
        result = doc.copy()
        result['bm25_score'] = float(bm25_scores[i])
        result['bm25_normalized'] = float(bm25_normalized[i])
        result['cosine_score'] = float(cosine_scores[i])
        result['cosine_normalized'] = float(cosine_normalized[i])
        result['hybrid_score'] = float(hybrid_scores[i])
        results.append(result)
    
    # Sort by hybrid score and return top-k
    results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return results[:top_k]

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range using min-max normalization."""
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.ones_like(scores) * 0.5  # All equal scores
    
    return (scores - min_score) / (max_score - min_score)

def display_results(query: str, results: List[Dict[str, Any]], lambda_param: float):
    """Display hybrid search results with all scoring details."""
    print(f"\n🔍 Query: '{query}' (λ = {lambda_param})")
    print("=" * 80)
    
    if not results:
        print("No results found.")
        return
    
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc['title']}")
        print(f"    Authors: {', '.join(doc.get('authors', ['Unknown']))}")
        print(f"    Venue: {doc.get('venue', 'Unknown')} ({doc.get('year', 'Unknown')})")
        print(f"    📊 Scores:")
        print(f"        BM25: {doc['bm25_score']:.4f} (norm: {doc['bm25_normalized']:.4f})")
        print(f"        Cosine: {doc['cosine_score']:.4f} (norm: {doc['cosine_normalized']:.4f})")
        print(f"        🎯 Hybrid: {doc['hybrid_score']:.4f}")
        print(f"    Tags: {', '.join(doc.get('tags', []))}")
        print(f"    Abstract: {doc.get('abstract', '')[:150]}...")

def get_test_queries() -> List[str]:
    """Get 10 hand-picked test queries for lambda tuning."""
    return [
        "front-door criterion",
        "exclusion restriction", 
        "identification strategies",
        "causal graphs",
        "heterogeneous effects",
        "natural experiments",
        "mediation analysis",
        "policy evaluation",
        "treatment assignment",
        "confounding adjustment"
    ]

def tune_lambda(
    docs: List[Dict[str, Any]], 
    bm25: BM25Okapi, 
    faiss_index: faiss.Index, 
    embedding_model: SentenceTransformer,
    lambda_values: List[float] = None
):
    """Tune lambda parameter using test queries."""
    if lambda_values is None:
        lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    test_queries = get_test_queries()
    
    print(f"\n🎯 Tuning λ parameter with {len(test_queries)} test queries...")
    print(f"Testing λ values: {lambda_values}")
    
    for lambda_val in lambda_values:
        print(f"\n" + "="*60)
        print(f"🔬 Testing λ = {lambda_val}")
        print("="*60)
        
        for query in test_queries[:3]:  # Show first 3 queries for each lambda
            results = hybrid_search(
                query, docs, bm25, faiss_index, embedding_model, 
                lambda_param=lambda_val, top_k=3
            )
            display_results(query, results, lambda_val)
            
        print(f"\n💡 λ = {lambda_val} Summary:")
        if lambda_val == 0.0:
            print("    Pure BM25 - Only exact word matching")
        elif lambda_val < 0.5:
            print("    BM25-heavy - Favors exact terms with some semantic understanding")
        elif lambda_val == 0.5:
            print("    Balanced - Equal weight to exact matching and semantic similarity")
        elif lambda_val < 1.0:
            print("    Embedding-heavy - Favors semantic similarity with some exact matching")
        else:
            print("    Embedding-dominant - Primarily semantic similarity")

def save_hybrid_index(
    bm25: BM25Okapi, 
    embeddings: np.ndarray,
    faiss_index: faiss.Index, 
    embedding_model: SentenceTransformer,
    docs: List[Dict[str, Any]], 
    artifacts_path: str
):
    """Save hybrid index components to disk."""
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Save BM25 and metadata
    hybrid_file = os.path.join(artifacts_path, "hybrid_index.pkl")
    with open(hybrid_file, 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'docs': docs,
            'embeddings': embeddings,
            'model_name': embedding_model._modules['0'].auto_model.config.name_or_path,
            'metadata': {
                'num_docs': len(docs),
                'embedding_dim': embeddings.shape[1],
                'index_type': 'Hybrid_BM25_Embeddings'
            }
        }, f)
    
    # Save FAISS index separately
    faiss_file = os.path.join(artifacts_path, "faiss_index.bin")
    faiss.write_index(faiss_index, faiss_file)
    
    print(f"✅ Hybrid index saved to {hybrid_file}")
    print(f"✅ FAISS index saved to {faiss_file}")

def load_hybrid_index(artifacts_path: str):
    """Load hybrid index from disk."""
    hybrid_file = os.path.join(artifacts_path, "hybrid_index.pkl")
    faiss_file = os.path.join(artifacts_path, "faiss_index.bin")
    
    if not os.path.exists(hybrid_file) or not os.path.exists(faiss_file):
        raise FileNotFoundError(f"Hybrid index files not found in {artifacts_path}")
    
    # Load metadata and BM25
    with open(hybrid_file, 'rb') as f:
        data = pickle.load(f)
    
    # Load FAISS index
    faiss_index = faiss.read_index(faiss_file)
    
    # Reload embedding model
    model_name = data.get('model_name', 'all-mpnet-base-v2')
    embedding_model = SentenceTransformer(model_name)
    
    return data['bm25'], data['docs'], faiss_index, embedding_model

def main():
    parser = argparse.ArgumentParser(description='Hybrid BM25 + Embeddings Search System')
    parser.add_argument('--build', action='store_true', help='Build hybrid index')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--tune-lambda', action='store_true', help='Tune lambda parameter')
    parser.add_argument('--lambda-param', type=float, default=0.5, help='Lambda parameter for hybrid scoring')
    parser.add_argument('--corpus', type=str, default='data/corpus.jsonl', help='Path to corpus JSONL file')
    parser.add_argument('--artifacts', type=str, default='artifacts', help='Path to artifacts directory')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2', help='Sentence transformer model')
    parser.add_argument('--interactive', action='store_true', help='Interactive search mode')
    
    args = parser.parse_args()
    
    # Ensure we're in project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if args.build:
        # Build hybrid index
        print("🔨 Building Hybrid BM25 + Embeddings Index...")
        docs = load_corpus(args.corpus)
        
        # Build BM25 index
        bm25 = build_bm25_index(docs)
        
        # Build embedding index
        embeddings, faiss_index, embedding_model = build_embedding_index(docs, args.model)
        
        # Save everything
        save_hybrid_index(bm25, embeddings, faiss_index, embedding_model, docs, args.artifacts)
        
        # Quick test
        print("\n🧪 Quick test with sample query...")
        test_query = "causal inference"
        results = hybrid_search(test_query, docs, bm25, faiss_index, embedding_model, getattr(args, 'lambda_param'), 3)
        display_results(test_query, results, getattr(args, 'lambda_param'))
        
    elif args.tune_lambda:
        # Lambda tuning mode
        try:
            bm25, docs, faiss_index, embedding_model = load_hybrid_index(args.artifacts)
            tune_lambda(docs, bm25, faiss_index, embedding_model)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Run with --build first to create the hybrid index.")
            sys.exit(1)
            
    elif args.search:
        # Search mode
        try:
            bm25, docs, faiss_index, embedding_model = load_hybrid_index(args.artifacts)
            results = hybrid_search(args.search, docs, bm25, faiss_index, embedding_model, getattr(args, 'lambda_param'), args.top_k)
            display_results(args.search, results, getattr(args, 'lambda_param'))
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Run with --build first to create the hybrid index.")
            sys.exit(1)
            
    elif args.interactive:
        # Interactive mode
        try:
            bm25, docs, faiss_index, embedding_model = load_hybrid_index(args.artifacts)
            print("🎯 Interactive Hybrid Search (type 'quit' to exit)")
            print(f"Current λ = {getattr(args, 'lambda_param')} (use --lambda-param to change)")
            print("Examples: 'identification strategies', 'causal graphs', 'policy evaluation'")
            
            while True:
                query = input("\n🔍 Enter query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    results = hybrid_search(query, docs, bm25, faiss_index, embedding_model, getattr(args, 'lambda_param'), args.top_k)
                    display_results(query, results, getattr(args, 'lambda_param'))
                    
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Run with --build first to create the hybrid index.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
