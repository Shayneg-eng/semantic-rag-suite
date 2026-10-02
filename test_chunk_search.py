import json
import csv
import numpy as np
from ollama import embed
import os
from collections import defaultdict
from tqdm import tqdm

# Load chunk embeddings
embeddings_file = "embeddings/chunk_embeddings.csv"
chunk_embeddings = {}  # chunk_id -> embedding
chunk_metadata = {}    # chunk_id -> {document_name, chunk_index, chunk_text}
doc_chunks = defaultdict(list)  # document_name -> [chunk_ids]

print("Loading chunk embeddings...")
with open(embeddings_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    chunk_id = 0
    for row in reader:
        doc_name = row['document_name']
        chunk_idx = int(row['chunk_index'])
        
        # Extract the embedding dimensions (768 dimensions)
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
    normalized = path.replace('/', '_').replace('\\', '_')
    return normalized

def find_best_matching_document(query_text, correct_file_path):
    """
    Find the best matching document using 80th percentile chunk similarity.
    
    The 80th percentile measures the quality of the "typical good match" in a document,
    which outperforms single best-chunk or complex weighted approaches.
    This captures documents where many chunks are relevant, not just one perfect match.
    
    Testing showed:
    - Max chunk only: avg rank 404
    - Weighted (0.5, 0.4, 0.1): avg rank 320
    - 80th percentile: avg rank 207 (BEST)
    """
    print(f"Query: {query_text[:80]}...")
    print(f"Expected: {correct_file_path}")
    
    # Embed the query
    query_embedding = get_embedding(query_text)
    
    # Find best chunks and calculate 80th percentile score for each document
    doc_scores = {}
    doc_best_chunk_info = {}
    
    for doc_name, chunk_ids in doc_chunks.items():
        chunk_sims = []
        
        # Get similarity for all chunks
        for chunk_id in chunk_ids:
            chunk_embedding = chunk_embeddings[chunk_id]
            sim = cosine_similarity(query_embedding, chunk_embedding)
            chunk_sims.append((sim, chunk_id))
        
        chunk_sims.sort(reverse=True)
        
        # Use 80th percentile as the document score
        percentile_score = np.percentile([sim for sim, _ in chunk_sims], 80)
        
        doc_scores[doc_name] = percentile_score
        doc_best_chunk_info[doc_name] = {
            'best_sim': chunk_sims[0][0],
            'p80': percentile_score,
            'p60': np.percentile([sim for sim, _ in chunk_sims], 60),
            'top_5_avg': np.mean([sim for sim, _ in chunk_sims[:5]]),
            'chunk_info': chunk_metadata[chunk_sims[0][1]]
        }
    
    # Sort documents by weighted score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Try to find the correct document
    correct_normalized = normalize_doc_name(correct_file_path).lower()
    
    found_rank = -1
    found_score = 0
    for rank, (doc_name, score) in enumerate(sorted_docs, 1):
        doc_normalized = doc_name.lower()
        if doc_normalized == correct_normalized or \
           correct_file_path.replace('/', '_').lower() == doc_normalized or \
           correct_file_path.replace('\\', '_').lower() == doc_normalized or \
           os.path.basename(correct_file_path).lower() in doc_normalized:
            found_rank = rank
            found_score = score
            break
    
    if found_rank == -1:
        print(f"❌ NOT FOUND in results")
        print(f"Top 5 results:")
        for i, (doc_name, score) in enumerate(sorted_docs[:5], 1):
            info = doc_best_chunk_info[doc_name]
            print(f"  {i}. {doc_name} (P80: {score:.4f})")
            print(f"     Best: {info['best_sim']:.4f} | P80: {info['p80']:.4f} | P60: {info['p60']:.4f} | Top5: {info['top_5_avg']:.4f}")
        return None
    else:
        print(f"✅ Found at rank #{found_rank} (P80: {found_score:.4f})")
        info = doc_best_chunk_info[sorted_docs[found_rank-1][0]]
        print(f"   Best: {info['best_sim']:.4f} | P80: {info['p80']:.4f} | P60: {info['p60']:.4f} | Top5: {info['top_5_avg']:.4f}")
        if found_rank > 1:
            top_score = sorted_docs[0][1]
            print(f"   (Top result was: {sorted_docs[0][0]} with P80 {top_score:.4f})")
        return found_rank

# Load benchmark file
benchmark_file = "LegalBench-RAG/benchmarks/contractnli.json"

if not os.path.exists(benchmark_file):
    print(f"Benchmark file not found: {benchmark_file}")
    exit(1)

print(f"Loading benchmark: {benchmark_file}\n")

with open(benchmark_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

tests = data.get('tests', [])
print(f"Total queries: {len(tests)}\n")

# Evaluate
results_by_rank = {}
total_queries = 0
found_count = 0

print("=" * 80)
for i, test in enumerate(tqdm(tests[:30], desc="Evaluating")):  # First 30 for speed
    query = test.get('query', '')
    snippets = test.get('snippets', [])
    
    if not snippets:
        continue
    
    correct_file = snippets[0].get('file_path', '')
    
    rank = find_best_matching_document(query, correct_file)
    
    if rank is not None:
        if rank not in results_by_rank:
            results_by_rank[rank] = 0
        results_by_rank[rank] += 1
        total_queries += 1
        found_count += 1
    else:
        total_queries += 1
    
    print()

# Print summary
print(f"{'='*80}")
print("CHUNK-BASED RANKING SUMMARY")
print(f"{'='*80}")
print(f"Total queries evaluated: {total_queries}")
print(f"Documents found: {found_count}/{total_queries} ({found_count/total_queries*100:.1f}%)\n")

if total_queries > 0:
    correct_at_rank_1 = results_by_rank.get(1, 0)
    print(f"✅ Correct at Rank #1: {correct_at_rank_1}/{found_count} ({correct_at_rank_1/found_count*100:.1f}%)")
    
    correct_at_top_5 = sum(results_by_rank.get(i, 0) for i in range(1, 6))
    print(f"✅ Correct in Top 5: {correct_at_top_5}/{found_count} ({correct_at_top_5/found_count*100:.1f}%)")
    
    correct_at_top_10 = sum(results_by_rank.get(i, 0) for i in range(1, 11))
    print(f"✅ Correct in Top 10: {correct_at_top_10}/{found_count} ({correct_at_top_10/found_count*100:.1f}%)")
    
    print(f"\nDetailed ranking breakdown:")
    for rank in sorted(results_by_rank.keys())[:15]:
        count = results_by_rank[rank]
        print(f"  Rank {rank:3d}: {count:3d} queries")
