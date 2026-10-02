import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from search_and_rank import get_query_embedding, rank_document
import pickle
import numpy as np

# Load vector database
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'vector_db.pkl')

print("Loading vector database...")
with open(VECTOR_DB_PATH, 'rb') as f:
    vector_db = pickle.load(f)

doc_names_list = sorted(list(vector_db.keys()))
doc_embeddings_array = np.array([np.array(vector_db[name]['embedding'], dtype=np.float32) 
                                 for name in doc_names_list], dtype=np.float32)

print(f"Loaded {len(vector_db)} documents\n")

# Test case
query = "confidentiality non-disclosure agreement"
doc_name = "corpus_chunks\contractnli_nda_form_motorola___chunk_0001.txt"
doc_embedding = np.array(vector_db[doc_name]['embedding'], dtype=np.float32)

print("=" * 80)
print("TEST: Search and Rank")
print("=" * 80)
print(f"\nQuery: {query}")
print(f"Document: {doc_name}")

# Get query embedding
print("\nProcessing...")
query_embedding = get_query_embedding(query)

# Rank document
rank, target_distance, all_distances = rank_document(query_embedding, doc_embedding, doc_embeddings_array)
target_similarity = 1 - target_distance

# Get top-10
sorted_indices = np.argsort(all_distances)
top_10_docs = [doc_names_list[i] for i in sorted_indices[:10]]
top_10_sims = [1 - all_distances[i] for i in sorted_indices[:10]]

# Results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nRanking Position: #{rank} out of {len(vector_db)} documents")
print(f"Similarity Score: {target_similarity:.4f}")

print(f"\nTop-10 Retrieved Documents:")
for rank_pos, (doc, sim) in enumerate(zip(top_10_docs, top_10_sims), 1):
    is_target = " <-- YOUR DOCUMENT" if doc == doc_name else ""
    print(f"  {rank_pos:2d}. {doc[:70]:70s} (sim: {sim:.4f}){is_target}")

if rank > 10:
    print(f"\n  ... ({rank - 10} documents ranked higher than yours)")
