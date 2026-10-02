"""
Normal RAG Baseline vs Directional Shift Approach

Compares standard embedding cosine similarity (normal RAG) 
against directional shift approach on the same test cases.
"""

import ollama
import numpy as np
from typing import List

EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> List[float]:
    """Get embedding from ollama."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        return response['embeddings'][0]
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_documents():
    """Load all 10 documents from files."""
    docs = {}
    for i in range(1, 11):
        try:
            with open(f"DOC_{i}.txt", "r") as f:
                docs[i] = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: DOC_{i}.txt not found")
    return docs


def baseline_rag_score(query: str, document: str) -> float:
    """Standard RAG: cosine similarity of raw embeddings."""
    q_emb = get_embedding(query)
    d_emb = get_embedding(document)
    
    if q_emb is None or d_emb is None:
        return None
    
    q_arr = np.array(q_emb)
    d_arr = np.array(d_emb)
    
    norm_q = np.linalg.norm(q_arr)
    norm_d = np.linalg.norm(d_arr)
    
    if norm_q == 0 or norm_d == 0:
        return 0.0
    
    return float(np.dot(q_arr, d_arr) / (norm_q * norm_d))


def run_comparison():
    """Compare baseline RAG against all documents."""
    
    # Load documents
    docs = load_documents()
    if len(docs) < 10:
        print(f"Error: Expected 10 documents, found {len(docs)}")
        return
    
    # Test queries
    queries = [
        {
            "name": "OAuth2",
            "text": "OAuth2 uses access tokens instead of credentials, significantly improving security. When implementing OAuth2, the authentication system must maintain proper token management practices.",
            "relevant_doc": 1,
        },
        {
            "name": "Microservices",
            "text": "Microservices architecture decomposes applications into independent services that communicate through well-defined APIs. Each service handles specific business capabilities and can be deployed independently.",
            "relevant_doc": 3,
        },
        {
            "name": "Rate Limiting",
            "text": "Rate limiting controls request frequency, protecting services from overload. Token bucket algorithms maintain available requests and regenerate at fixed rates.",
            "relevant_doc": 9,
        },
        {
            "name": "Logging",
            "text": "Comprehensive logging provides visibility into system behavior. Structured logging uses consistent formats like JSON to record events with context.",
            "relevant_doc": 5,
        },
        {
            "name": "Error Handling",
            "text": "Robust error handling distinguishes between transient failures and permanent failures. Circuit breaker patterns protect services from cascading failures.",
            "relevant_doc": 10,
        },
    ]
    
    print(f"\n{'='*80}")
    print("BASELINE RAG EVALUATION - Cosine Similarity")
    print(f"{'='*80}\n")
    
    results = []
    
    for query_info in queries:
        print(f"Query: {query_info['name']}")
        print(f"{'─'*80}")
        
        query = query_info['text']
        relevant_doc_idx = query_info['relevant_doc']
        
        # Score all 10 documents
        scores = []
        for doc_idx in range(1, 11):
            score = baseline_rag_score(query, docs[doc_idx])
            scores.append((doc_idx, score, "RELEVANT" if doc_idx == relevant_doc_idx else "IRRELEVANT"))
        
        # Sort by score (descending)
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Find rank of relevant document
        relevant_rank = None
        for rank, (doc_idx, score, label) in enumerate(scores_sorted, 1):
            if doc_idx == relevant_doc_idx:
                relevant_rank = rank
                relevant_score = score
                break
        
        # Show top 5
        print(f"Top 5 ranked documents:")
        for rank, (doc_idx, score, label) in enumerate(scores_sorted[:5], 1):
            marker = "✓" if doc_idx == relevant_doc_idx else " "
            print(f"  {rank}. DOC_{doc_idx:2d}: {score:.4f} {marker} {label}")
        
        print(f"\nRelevant document (DOC_{relevant_doc_idx}) rank: {relevant_rank}/10")
        print(f"Relevant score: {relevant_score:.4f}")
        
        # Calculate gap between relevant and top irrelevant
        top_irrelevant_score = max(s for idx, s, label in scores if idx != relevant_doc_idx)
        gap = relevant_score - top_irrelevant_score
        
        print(f"Top irrelevant score: {top_irrelevant_score:.4f}")
        print(f"Gap: {gap:+.4f}\n")
        
        results.append({
            'name': query_info['name'],
            'relevant_rank': relevant_rank,
            'relevant_score': relevant_score,
            'top_irrelevant_score': top_irrelevant_score,
            'gap': gap,
        })
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY - Baseline RAG Performance")
    print(f"{'='*80}\n")
    
    print(f"Results:")
    for r in results:
        rank_str = f"#{r['relevant_rank']}" if r['relevant_rank'] else "NOT FOUND"
        print(f"  {r['name']:15s}: Rank {rank_str:4s}, Score {r['relevant_score']:.4f}, Gap {r['gap']:+.4f}")
    
    avg_rank = np.mean([r['relevant_rank'] for r in results])
    avg_gap = np.mean([r['gap'] for r in results])
    
    print(f"\nAverage rank: {avg_rank:.1f}/10")
    print(f"Average gap: {avg_gap:+.4f}")
    
    rank_1_count = sum(1 for r in results if r['relevant_rank'] == 1)
    rank_top5_count = sum(1 for r in results if r['relevant_rank'] <= 5)
    
    print(f"Rank #1: {rank_1_count}/5")
    print(f"Top 5: {rank_top5_count}/5")
    
    print(f"\n{'CONCLUSION':-^80}")
    print(f"\nBaseline RAG (embedding cosine similarity):")
    if rank_1_count >= 4:
        print(f"✓ EXCELLENT - Relevant docs rank #1 most of the time")
    elif rank_top5_count >= 4:
        print(f"✓ GOOD - Relevant docs in top 5 most of the time")
    elif avg_rank <= 5:
        print(f"⚠ MODERATE - Average rank is {avg_rank:.1f}")
    else:
        print(f"✗ POOR - Relevant docs often ranked low")
    
    print(f"\nCompare to directional shift approach:")
    print(f"  Directional shift average gap: +0.1387")
    print(f"  Baseline RAG average gap: {avg_gap:+.4f}")
    
    if avg_gap > 0.1387:
        print(f"\n✗ Baseline is BETTER than directional shift approach")
        print(f"  Use standard embedding-based RAG, no need for transformations")
    else:
        print(f"\n✓ Directional shift provides ADDITIONAL IMPROVEMENT")
        print(f"  Gap improvement: {(0.1387 - avg_gap) / avg_gap * 100:+.1f}%")


if __name__ == "__main__":
    run_comparison()
