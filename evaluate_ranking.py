import json
import csv
import numpy as np
from ollama import embed
import os
from pathlib import Path
from tqdm import tqdm

# Load document embeddings
embeddings_file = "embeddings/document_embeddings.csv"
doc_embeddings = {}
doc_paths = {}

print("Loading document embeddings...")
with open(embeddings_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_name = row['document_name']
        doc_path = row['document_path']
        
        # Extract the embedding dimensions (768 dimensions)
        embedding = np.array([float(row[f'dim_{i}']) for i in range(768)])
        
        doc_embeddings[doc_name] = embedding
        doc_paths[doc_name] = doc_path

print(f"Loaded {len(doc_embeddings)} documents\n")

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
    # Remove leading dataset prefix if present
    name = os.path.basename(path)
    # Replace forward and back slashes with underscores
    normalized = path.replace('/', '_').replace('\\', '_')
    return normalized

def find_correct_document_rank(query, correct_file_path):
    """Find what rank the correct document appears at"""
    print(f"\nQuery: {query[:80]}...")
    print(f"Expected: {correct_file_path}")
    
    # Embed the query
    query_embedding = get_embedding(query)
    
    # Calculate similarity to all documents
    similarities = []
    for doc_name, doc_embedding in doc_embeddings.items():
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_name, sim, doc_paths[doc_name]))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Try to find the correct document
    correct_normalized = normalize_doc_name(correct_file_path)
    
    found_rank = -1
    for rank, (doc_name, sim, path) in enumerate(similarities, 1):
        # Check if this is the correct document
        doc_normalized = doc_name.lower()
        correct_lower = correct_normalized.lower()
        
        # Multiple matching strategies
        if doc_normalized == correct_lower or \
           correct_file_path.replace('/', '_').lower() == doc_normalized or \
           correct_file_path.replace('\\', '_').lower() == doc_normalized or \
           os.path.basename(correct_file_path).lower() in doc_normalized or \
           doc_normalized in correct_file_path.lower():
            found_rank = rank
            found_sim = sim
            break
    
    if found_rank == -1:
        print(f"❌ NOT FOUND in top results")
        print(f"Top 5 results:")
        for i, (doc_name, sim, path) in enumerate(similarities[:5], 1):
            print(f"  {i}. {doc_name} (sim: {sim:.4f})")
        return None
    else:
        print(f"✅ Found at rank #{found_rank} (similarity: {found_sim:.4f})")
        print(f"   Retrieved: {similarities[found_rank-1][0]}")
        if found_rank > 1:
            print(f"   (Top result was: {similarities[0][0]} with sim {similarities[0][1]:.4f})")
        return found_rank

# Load benchmark files
benchmark_files = [
    "LegalBench-RAG/benchmarks/contractnli.json",
    "LegalBench-RAG/benchmarks/cuad.json",
    "LegalBench-RAG/benchmarks/maud.json",
    "LegalBench-RAG/benchmarks/privacy_qa.json"
]

results_by_rank = {}
total_queries = 0

for bench_file in benchmark_files:
    if not os.path.exists(bench_file):
        print(f"Skipping {bench_file} (not found)")
        continue
    
    print(f"\n{'='*80}")
    print(f"Processing {os.path.basename(bench_file)}")
    print(f"{'='*80}")
    
    with open(bench_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tests = data.get('tests', [])
    print(f"Total queries: {len(tests)}")
    
    for test in tqdm(tests[:20], desc=f"Evaluating"):  # First 20 for speed
        query = test.get('query', '')
        snippets = test.get('snippets', [])
        
        if not snippets:
            continue
        
        correct_file = snippets[0].get('file_path', '')
        
        rank = find_correct_document_rank(query, correct_file)
        
        if rank is not None:
            if rank not in results_by_rank:
                results_by_rank[rank] = 0
            results_by_rank[rank] += 1
            total_queries += 1

# Print summary
print(f"\n\n{'='*80}")
print("RANKING SUMMARY")
print(f"{'='*80}")
print(f"Total queries evaluated: {total_queries}\n")

if total_queries > 0:
    correct_at_rank_1 = results_by_rank.get(1, 0)
    print(f"✅ Correct at Rank #1: {correct_at_rank_1}/{total_queries} ({correct_at_rank_1/total_queries*100:.1f}%)")
    
    correct_at_top_5 = sum(results_by_rank.get(i, 0) for i in range(1, 6))
    print(f"✅ Correct in Top 5: {correct_at_top_5}/{total_queries} ({correct_at_top_5/total_queries*100:.1f}%)")
    
    print(f"\nDetailed ranking breakdown:")
    for rank in sorted(results_by_rank.keys())[:20]:
        count = results_by_rank[rank]
        print(f"  Rank {rank:3d}: {count:3d} queries")
