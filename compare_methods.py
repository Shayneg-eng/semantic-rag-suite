import json
import csv
import numpy as np
from ollama import embed
import os
from collections import defaultdict

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

def evaluate_method(method_name, score_func, queries_list):
    """Evaluate a scoring method on the given queries"""
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
                chunk_sims.append(sim)
            
            # Call the scoring function
            score = score_func(chunk_sims)
            doc_scores[doc_name] = score
        
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
    
    avg_rank = np.mean(results['ranks']) if results['ranks'] else 999
    print(f"{method_name:<40} | Rank#1: {results['rank_1']:<2} | Top5: {results['top_5']:<2} | Top10: {results['top_10']:<2} | Avg: {avg_rank:7.1f}")
    
    return results

# Load benchmark
benchmark_file = "LegalBench-RAG/benchmarks/contractnli.json"
with open(benchmark_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

tests = data.get('tests', [])
queries_list = []
for test in tests[:50]:
    query = test.get('query', '')
    snippets = test.get('snippets', [])
    if snippets:
        correct_file = snippets[0].get('file_path', '')
        queries_list.append((query, correct_file))

print(f"Testing with {len(queries_list)} queries\n")

# Define different scoring methods
def method_best_only(sims):
    """Just use the best chunk"""
    return max(sims)

def method_top_k_mean(k):
    """Mean of top K chunks"""
    def score_func(sims):
        sorted_sims = sorted(sims, reverse=True)
        return np.mean(sorted_sims[:k])
    return score_func

def method_top_k_weighted(k, decay=0.9):
    """Weighted sum of top K chunks (with decay)"""
    def score_func(sims):
        sorted_sims = sorted(sims, reverse=True)
        score = 0
        for i, sim in enumerate(sorted_sims[:k]):
            weight = decay ** i
            score += sim * weight
        return score / k
    return score_func

def method_percentile(p):
    """Use the Pth percentile as score"""
    def score_func(sims):
        return np.percentile(sims, p)
    return score_func

def method_count_above_threshold(thresh):
    """Count chunks above threshold, weighted by best chunk"""
    def score_func(sims):
        best = max(sims)
        count = sum(1 for s in sims if s > thresh)
        return best * (1 + count * 0.1)
    return score_func

# Test methods
print("METHOD                                   | Rank#1 | Top5 | Top10 | Avg Rank")
print("=" * 80)

evaluate_method("Best chunk only", method_best_only, queries_list)
evaluate_method("Top 3 chunks (mean)", method_top_k_mean(3), queries_list)
evaluate_method("Top 5 chunks (mean)", method_top_k_mean(5), queries_list)
evaluate_method("Top 10 chunks (mean)", method_top_k_mean(10), queries_list)
evaluate_method("Top 3 chunks (weighted)", method_top_k_weighted(3, 0.9), queries_list)
evaluate_method("Top 5 chunks (weighted)", method_top_k_weighted(5, 0.9), queries_list)
evaluate_method("Top 10 chunks (weighted)", method_top_k_weighted(10, 0.9), queries_list)
evaluate_method("Top 10 chunks (weighted, 0.8 decay)", method_top_k_weighted(10, 0.8), queries_list)
evaluate_method("90th percentile", method_percentile(90), queries_list)
evaluate_method("75th percentile", method_percentile(75), queries_list)
evaluate_method("Count >0.5 threshold", method_count_above_threshold(0.5), queries_list)
evaluate_method("Count >0.6 threshold", method_count_above_threshold(0.6), queries_list)
evaluate_method("Count >0.65 threshold", method_count_above_threshold(0.65), queries_list)
