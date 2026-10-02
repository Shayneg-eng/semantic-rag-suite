import os
import csv
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'query_document_pairs.csv')
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'vector_db.pkl')

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors (1 = identical, -1 = opposite)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def main():
    """Analyze retrieval performance in detail."""
    
    print("Detailed Retrieval Analysis")
    print("=" * 80)
    
    # Load vector database
    with open(VECTOR_DB_PATH, 'rb') as f:
        vector_db = pickle.load(f)
    
    # Load CSV
    queries = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    
    # Analyze each query
    similarities = []
    positions = []
    query_performance = []
    
    print(f"\nAnalyzing {len(queries)} queries...\n")
    
    for idx, row in enumerate(queries):
        query_embedding = eval(row['query_embedding'])
        correct_doc = row['correct_document']
        query_text = row['query']
        
        # Get similarity to correct document
        if correct_doc in vector_db:
            doc_embedding = vector_db[correct_doc]['embedding']
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append(sim)
            
            # Compute position
            query_embedding = np.array(query_embedding, dtype=np.float32)
            doc_embeddings_array = np.array([np.array(vector_db[name]['embedding'], dtype=np.float32) 
                                             for name in vector_db.keys()], dtype=np.float32)
            doc_names_list = sorted(list(vector_db.keys()))
            
            query_norm = np.linalg.norm(query_embedding)
            dot_products = doc_embeddings_array @ query_embedding
            doc_norms = np.linalg.norm(doc_embeddings_array, axis=1)
            
            valid = doc_norms > 0
            distances = np.ones(len(doc_norms))
            distances[valid] = 1 - (dot_products[valid] / (doc_norms[valid] * query_norm))
            
            sorted_indices = np.argsort(distances)
            sorted_docs = [doc_names_list[i] for i in sorted_indices]
            position = sorted_docs.index(correct_doc) + 1
            positions.append(position)
            
            query_performance.append({
                'query': query_text,
                'correct_doc': correct_doc,
                'similarity': sim,
                'position': position
            })
    
    # Statistics
    print("SIMILARITY ANALYSIS")
    print("=" * 80)
    similarities = np.array(similarities)
    print(f"Query-Document Similarity Statistics:")
    print(f"  Mean:      {np.mean(similarities):.4f}")
    print(f"  Median:    {np.median(similarities):.4f}")
    print(f"  Std Dev:   {np.std(similarities):.4f}")
    print(f"  Min:       {np.min(similarities):.4f}")
    print(f"  Max:       {np.max(similarities):.4f}")
    print(f"  25th pct:  {np.percentile(similarities, 25):.4f}")
    print(f"  75th pct:  {np.percentile(similarities, 75):.4f}")
    
    # Correlation between similarity and position
    positions = np.array(positions)
    correlation = np.corrcoef(similarities, positions)[0, 1]
    print(f"\nCorrelation (similarity vs position): {correlation:.4f}")
    if abs(correlation) > 0.7:
        print("  ✅ Strong correlation - higher similarity = better ranking")
    elif abs(correlation) > 0.4:
        print("  ⚠️  Moderate correlation")
    else:
        print("  ❌ Weak correlation - similarity and ranking are disconnected")
    
    # Best and worst performing queries
    print("\n" + "=" * 80)
    print("BEST PERFORMING QUERIES (Top-10)")
    print("=" * 80)
    sorted_by_position = sorted(query_performance, key=lambda x: x['position'])
    for i, q in enumerate(sorted_by_position[:10], 1):
        print(f"{i}. Position: {q['position']:5d} | Similarity: {q['similarity']:7.4f}")
        print(f"   Query: {q['query'][:70]}")
        print(f"   Doc:   {q['correct_doc'][:70]}")
        print()
    
    print("=" * 80)
    print("WORST PERFORMING QUERIES (Bottom-10)")
    print("=" * 80)
    for i, q in enumerate(sorted_by_position[-10:], 1):
        print(f"{i}. Position: {q['position']:5d} | Similarity: {q['similarity']:7.4f}")
        print(f"   Query: {q['query'][:70]}")
        print(f"   Doc:   {q['correct_doc'][:70]}")
        print()
    
    # Similarity distribution
    print("=" * 80)
    print("SIMILARITY DISTRIBUTION")
    print("=" * 80)
    ranges = [
        (-1.0, -0.5, "Opposite"),
        (-0.5, -0.0, "Dissimilar"),
        (0.0, 0.2, "Very Low"),
        (0.2, 0.4, "Low"),
        (0.4, 0.6, "Medium"),
        (0.6, 0.8, "High"),
        (0.8, 1.0, "Very High"),
    ]
    
    for min_val, max_val, label in ranges:
        count = np.sum((similarities >= min_val) & (similarities < max_val))
        pct = 100 * count / len(similarities)
        bar = "█" * int(pct / 2)
        print(f"{label:12s} [{min_val:5.1f}, {max_val:5.1f}): {count:4d} ({pct:5.1f}%) {bar}")
    
    # Position vs Similarity buckets
    print("\n" + "=" * 80)
    print("POSITION STATISTICS BY SIMILARITY RANGE")
    print("=" * 80)
    
    for min_val, max_val, label in ranges:
        mask = (similarities >= min_val) & (similarities < max_val)
        if np.sum(mask) > 0:
            positions_in_range = positions[mask]
            print(f"\n{label} similarity:")
            print(f"  Count:  {np.sum(mask)}")
            print(f"  Avg Position: {np.mean(positions_in_range):.0f}")
            print(f"  Median Position: {np.median(positions_in_range):.0f}")
            print(f"  Top-10 Retrieval: {100*np.sum(positions_in_range <= 10) / len(positions_in_range):.1f}%")

if __name__ == '__main__':
    main()
