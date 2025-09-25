#!/usr/bin/env python3
"""
Day 3: BM25 Baseline Implementation
Build BM25 index over document abstracts with CLI testing interface.
"""

import json
import pickle
import os
import argparse
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

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
    
    # Tokenize abstracts (simple whitespace + lowercase)
    tokenized_abstracts = []
    for doc in docs:
        abstract = doc.get('abstract', '')
        tokens = abstract.lower().split()
        tokenized_abstracts.append(tokens)
        
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_abstracts)
    return bm25

def search_bm25(bm25: BM25Okapi, docs: List[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search BM25 index and return ranked results."""
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k results with scores
    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for doc, score in doc_scores[:top_k]:
        result = doc.copy()
        result['bm25_score'] = float(score)
        results.append(result)
    
    return results

def display_results(query: str, results: List[Dict[str, Any]]):
    """Display search results in a nice format."""
    print(f"\n🔍 Query: '{query}'")
    print("=" * 60)
    
    if not results:
        print("No results found.")
        return
    
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc['title']}")
        print(f"    Authors: {', '.join(doc.get('authors', ['Unknown']))}")
        print(f"    Venue: {doc.get('venue', 'Unknown')} ({doc.get('year', 'Unknown')})")
        print(f"    BM25 Score: {doc['bm25_score']:.4f}")
        print(f"    Tags: {', '.join(doc.get('tags', []))}")
        print(f"    Abstract: {doc.get('abstract', '')[:200]}...")
        print(f"    URL: {doc.get('url', '')}")

def save_index(bm25: BM25Okapi, docs: List[Dict[str, Any]], artifacts_path: str):
    """Save BM25 index and documents to disk."""
    os.makedirs(artifacts_path, exist_ok=True)
    index_file = os.path.join(artifacts_path, "bm25_index.pkl")
    
    with open(index_file, 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'docs': docs,
            'metadata': {
                'num_docs': len(docs),
                'index_type': 'BM25Okapi'
            }
        }, f)
    
    print(f"✅ BM25 index saved to {index_file}")

def load_index(artifacts_path: str):
    """Load BM25 index from disk."""
    index_file = os.path.join(artifacts_path, "bm25_index.pkl")
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    with open(index_file, 'rb') as f:
        data = pickle.load(f)
    
    return data['bm25'], data['docs']

def main():
    parser = argparse.ArgumentParser(description='BM25 Index Builder and Search CLI')
    parser.add_argument('--build', action='store_true', help='Build BM25 index')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--corpus', type=str, default='data/corpus.jsonl', help='Path to corpus JSONL file')
    parser.add_argument('--artifacts', type=str, default='artifacts', help='Path to artifacts directory')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--interactive', action='store_true', help='Interactive search mode')
    
    args = parser.parse_args()
    
    # Ensure we're in project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if args.build:
        # Build index
        print("🔨 Building BM25 index...")
        docs = load_corpus(args.corpus)
        bm25 = build_bm25_index(docs)
        save_index(bm25, docs, args.artifacts)
        
        # Test with sample queries
        print("\n🧪 Testing with sample queries...")
        test_queries = [
            "front-door criterion",
            "exclusion restriction", 
            "instrumental variables",
            "back-door adjustment"
        ]
        
        for query in test_queries:
            results = search_bm25(bm25, docs, query, top_k=3)
            display_results(query, results)
            
    elif args.search:
        # Search mode
        try:
            bm25, docs = load_index(args.artifacts)
            results = search_bm25(bm25, docs, args.search, args.top_k)
            display_results(args.search, results)
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Run with --build first to create the index.")
            sys.exit(1)
            
    elif args.interactive:
        # Interactive mode
        try:
            bm25, docs = load_index(args.artifacts)
            print("🎯 Interactive BM25 Search (type 'quit' to exit)")
            print("Examples: 'front-door criterion', 'exclusion restriction', 'causal inference'")
            
            while True:
                query = input("\n🔍 Enter query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    results = search_bm25(bm25, docs, query, args.top_k)
                    display_results(query, results)
                    
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Run with --build first to create the index.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
