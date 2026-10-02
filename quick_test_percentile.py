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
        chunk_metadata[chunk_id] = {'document_name': doc_name, 'chunk_index': chunk_idx}
        doc_chunks[doc_name].append(chunk_id)
        chunk_id += 1

print(f"Loaded {len(chunk_embeddings)} chunks from {len(doc_chunks)} documents\n")

def get_embedding(text):
    response = embed(model="nomic-embed-text", input=text)
    return np.array(response['embeddings'][0])

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def normalize_doc_name(path):
    return path.replace('/', '_').replace('\\', '_')

# Load benchmark
benchmark_file = "LegalBench-RAG/benchmarks/contractnli.json"
with open(benchmark_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

tests = data.get('tests', [])
queries_list = []
for test in tests:
    query = test.get('query', '')
    snippets = test.get('snippets', [])
    if snippets:
        correct_file = snippets[0].get('file_path', '')
        queries_list.append((query, correct_file))

print(f"Testing {len(queries_list)} queries using 80th percentile scoring...\n")

# Test 80th percentile method
rank_1 = 0
top_5 = 0
top_10 = 0
ranks = []

for i, (query_text, correct_file_path) in enumerate(queries_list):
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(queries_list)} queries...")
    
    query_embedding = get_embedding(query_text)
    doc_scores = {}
    
    for doc_name, chunk_ids in doc_chunks.items():
        chunk_sims = []
        for chunk_id in chunk_ids:
            chunk_embedding = chunk_embeddings[chunk_id]
            sim = cosine_similarity(query_embedding, chunk_embedding)
            chunk_sims.append(sim)
        
        if chunk_sims:
            doc_scores[doc_name] = np.percentile(chunk_sims, 80)
    
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    correct_normalized = normalize_doc_name(correct_file_path).lower()
    
    for rank, (doc_name, score) in enumerate(sorted_docs, 1):
        doc_normalized = doc_name.lower()
        if doc_normalized == correct_normalized or \
           correct_file_path.replace('/', '_').lower() == doc_normalized or \
           correct_file_path.replace('\\', '_').lower() == doc_normalized or \
           os.path.basename(correct_file_path).lower() in doc_normalized:
            ranks.append(rank)
            if rank == 1:
                rank_1 += 1
            if rank <= 5:
                top_5 += 1
            if rank <= 10:
                top_10 += 1
            break

avg_rank = np.mean(ranks)
print(f"\n{'='*80}")
print(f"RESULTS: 80th percentile scoring on {len(queries_list)} ContractNLI queries")
print(f"{'='*80}")
print(f"Rank #1:  {rank_1:3d}/{len(queries_list)} ({100*rank_1/len(queries_list):5.1f}%)")
print(f"Top 5:    {top_5:3d}/{len(queries_list)} ({100*top_5/len(queries_list):5.1f}%)")
print(f"Top 10:   {top_10:3d}/{len(queries_list)} ({100*top_10/len(queries_list):5.1f}%)")
print(f"Avg rank: {avg_rank:7.1f}")
print(f"{'='*80}")
