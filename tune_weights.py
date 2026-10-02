import json
import csv
import numpy as np
from ollama import embed
import os
from collections import defaultdict
from itertools import product

# Load chunk embeddings
embeddings_file = "embeddings/chunk_embeddings.csv"
chunk_embeddings = {}
chunk_metadata = {}
doc_chunks = defaultdict(list)

print("Loading chunk embeddings...")
with open(embeddings_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    chunk_id = 0
    for row in reader:
        doc_name = row['document_name']
        chunk_idx = int(row['chunk_index'])
        
        embedding = np.array([float(row[f'dim_{i}']) for i in range(768)])
        
        chunk_embeddings[chunk_id] = embedding
        chunk_metadata[chunk_id] = {
            'document_name': doc_name,
            'chunk_index': chunk_idx,
            'chunk_text': row['chunk_text']
        }
        doc_chunks[doc_name].append(chunk_id)
        chunk_id += 1

print(f"Loaded {len(chunk_embeddings)} chunks from {len(doc_chunks)} documents\n")

def get_embedding(text):
    """Get embedding for text using ollama"""
    response = embed(model="nomic-embed-text", input=text)
    return np.array(response['embeddings'][0])

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def normalize_doc_name(path):
    """Normalize document path to match embeddings naming"""
    return path.replace('/', '_').replace('\\', '_')

def evaluate_weights(w_best, w_ratio, w_top5, queries_list):
    """Evaluate a weight combination on the given queries"""
    results = {
        'rank_1': 0,
        'top_5': 0,
        'top_10': 0,
        'total': 0,
        'ranks': []
    }
    
    for query_text, correct_file_path in queries_list:
        query_embedding = get_embedding(query_text)
        
        doc_scores = {}
        
        for doc_name, chunk_ids in doc_chunks.items():
            chunk_sims = []
            
            for chunk_id in chunk_ids:
                chunk_embedding = chunk_embeddings[chunk_id]
                sim = cosine_similarity(query_embedding, chunk_embedding)
                chunk_sims.append((sim, chunk_id))
            
            chunk_sims.sort(reverse=True)
            
            best_sim = chunk_sims[0][0]
            similar_chunks_ratio = sum(1 for sim, _ in chunk_sims if sim > 0.5) / max(len(chunk_ids), 1)
            top_5_avg = np.mean([sim for sim, _ in chunk_sims[:5]])
            
            # Weighted score with current weights
            weighted_score = (best_sim * w_best) + (similar_chunks_ratio * w_ratio) + (top_5_avg * w_top5)
            doc_scores[doc_name] = weighted_score
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        correct_normalized = normalize_doc_name(correct_file_path).lower()
        
        for rank, (doc_name, score) in enumerate(sorted_docs, 1):
            doc_normalized = doc_name.lower()
            if doc_normalized == correct_normalized or \
               correct_file_path.replace('/', '_').lower() == doc_normalized or \
               correct_file_path.replace('\\', '_').lower() == doc_normalized or \
               os.path.basename(correct_file_path).lower() in doc_normalized:
                results['ranks'].append(rank)
                results['total'] += 1
                if rank == 1:
                    results['rank_1'] += 1
                if rank <= 5:
                    results['top_5'] += 1
                if rank <= 10:
                    results['top_10'] += 1
                break
    
    return results

# Load benchmark
benchmark_file = "LegalBench-RAG/benchmarks/contractnli.json"
with open(benchmark_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

tests = data.get('tests', [])
queries_list = []
for test in tests[:50]:  # Use 50 queries for tuning
    query = test.get('query', '')
    snippets = test.get('snippets', [])
    if snippets:
        correct_file = snippets[0].get('file_path', '')
        queries_list.append((query, correct_file))

print(f"Testing with {len(queries_list)} queries\n")

# Test different weight combinations - focus on ratio (chunk distribution)
# Generate combinations emphasizing ratio more
weight_combinations = []

# Aggressive ratio testing
for w_best in np.arange(0.3, 0.8, 0.1):
    for w_ratio in np.arange(0.1, 0.7, 0.1):
        w_top5 = 1.0 - w_best - w_ratio
        if w_top5 >= 0 and w_top5 <= 0.4:
            weight_combinations.append((round(w_best, 2), round(w_ratio, 2), round(w_top5, 2)))

# Remove duplicates
weight_combinations = list(set(weight_combinations))
weight_combinations.sort()

print(f"{'W_BEST':<7} {'W_RATIO':<7} {'W_TOP5':<7} | {'Rank#1':<6} {'Top5':<6} {'Top10':<6} | Avg Rank")
print("=" * 80)

best_result = None
best_weights = None

for w_best, w_ratio, w_top5 in weight_combinations:
    result = evaluate_weights(w_best, w_ratio, w_top5, queries_list)
    avg_rank = np.mean(result['ranks']) if result['ranks'] else 999
    
    score = result['top_5']  # Prioritize top 5
    
    marker = ""
    if best_result is None or score > best_result['top_5']:
        marker = " ← BEST"
        best_result = result
        best_weights = (w_best, w_ratio, w_top5)
    
    print(f"{w_best:<7.2f} {w_ratio:<7.2f} {w_top5:<7.2f} | "
          f"{result['rank_1']:<6}/{result['total']:<3} "
          f"{result['top_5']:<6}/{result['total']:<3} "
          f"{result['top_10']:<6}/{result['total']:<3} | "
          f"{avg_rank:6.1f}{marker}")

print("\n" + "=" * 80)
print(f"\nBEST WEIGHTS FOUND:")
print(f"  w_best:  {best_weights[0]}")
print(f"  w_ratio: {best_weights[1]}")
print(f"  w_top5:  {best_weights[2]}")
print(f"\nRESULTS:")
print(f"  Rank #1: {best_result['rank_1']}/{best_result['total']} ({best_result['rank_1']/best_result['total']*100:.1f}%)")
print(f"  Top 5:   {best_result['top_5']}/{best_result['total']} ({best_result['top_5']/best_result['total']*100:.1f}%)")
print(f"  Top 10:  {best_result['top_10']}/{best_result['total']} ({best_result['top_10']/best_result['total']*100:.1f}%)")
print(f"  Avg Rank: {np.mean(best_result['ranks']):.1f}")
