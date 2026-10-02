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
    print(f"{method_name:<50} | R1: {results['rank_1']:2} | T5: {results['top_5']:2} | T10: {results['top_10']:2} | Avg: {avg_rank:7.1f}")
    
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

# Test hybrid approaches
print("METHOD                                                 | R1 | T5 | T10 | Avg Rank")
print("=" * 90)

# Pure percentiles (baselines)
evaluate_method("BASELINE: 60th percentile", 
               lambda sims: np.percentile(sims, 60), queries_list)
evaluate_method("BASELINE: 80th percentile", 
               lambda sims: np.percentile(sims, 80), queries_list)

# Hybrid: percentile + best chunk
evaluate_method("HYBRID: 0.5*max + 0.5*80th percentile",
               lambda sims: 0.5 * max(sims) + 0.5 * np.percentile(sims, 80), queries_list)
evaluate_method("HYBRID: 0.6*max + 0.4*80th percentile",
               lambda sims: 0.6 * max(sims) + 0.4 * np.percentile(sims, 80), queries_list)
evaluate_method("HYBRID: 0.7*max + 0.3*80th percentile",
               lambda sims: 0.7 * max(sims) + 0.3 * np.percentile(sims, 80), queries_list)

# Hybrid: percentile + percentile
evaluate_method("HYBRID: 0.5*80th + 0.5*60th percentile",
               lambda sims: 0.5 * np.percentile(sims, 80) + 0.5 * np.percentile(sims, 60), queries_list)
evaluate_method("HYBRID: 0.6*80th + 0.4*60th percentile",
               lambda sims: 0.6 * np.percentile(sims, 80) + 0.4 * np.percentile(sims, 60), queries_list)
evaluate_method("HYBRID: 0.7*80th + 0.3*60th percentile",
               lambda sims: 0.7 * np.percentile(sims, 80) + 0.3 * np.percentile(sims, 60), queries_list)

# Three-way hybrid
evaluate_method("HYBRID: 0.5*max + 0.3*80th + 0.2*60th percentile",
               lambda sims: 0.5 * max(sims) + 0.3 * np.percentile(sims, 80) + 0.2 * np.percentile(sims, 60), queries_list)
evaluate_method("HYBRID: 0.4*max + 0.4*80th + 0.2*60th percentile",
               lambda sims: 0.4 * max(sims) + 0.4 * np.percentile(sims, 80) + 0.2 * np.percentile(sims, 60), queries_list)
evaluate_method("HYBRID: 0.3*max + 0.5*80th + 0.2*60th percentile",
               lambda sims: 0.3 * max(sims) + 0.5 * np.percentile(sims, 80) + 0.2 * np.percentile(sims, 60), queries_list)

# Quantile-based
evaluate_method("HYBRID: max(0.5*max, 0.8*80th percentile)",
               lambda sims: max(0.5 * max(sims), 0.8 * np.percentile(sims, 80)), queries_list)
