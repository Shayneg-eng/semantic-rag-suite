import os
import pickle
import numpy as np
from pathlib import Path
import ollama

# Configuration
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'vector_db.pkl')
EMBEDDING_MODEL = 'nomic-embed-text'

def get_query_embedding(query_text):
    """Get embedding for a search query using ollama."""
    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=query_text,
        )
        return response['embeddings'][0]
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

def get_document_embedding(doc_path):
    """Get embedding for a document file."""
    try:
        if not os.path.exists(doc_path):
            print(f"Error: Document file not found: {doc_path}")
            return None, None
        
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=content,
        )
        return response['embeddings'][0], content
    except Exception as e:
        print(f"Error embedding document: {e}")
        return None, None

def cosine_distance(a, b):
    """Compute cosine distance between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1 - (np.dot(a, b) / (norm_a * norm_b))

def rank_document(query_embedding, doc_embedding, all_doc_embeddings):
    """Find the rank position of a document embedding against all documents."""
    query_embedding = np.array(query_embedding, dtype=np.float32)
    doc_embedding = np.array(doc_embedding, dtype=np.float32)
    all_doc_embeddings = np.array(all_doc_embeddings, dtype=np.float32)
    
    # Compute distances
    query_norm = np.linalg.norm(query_embedding)
    dot_products = all_doc_embeddings @ query_embedding
    doc_norms = np.linalg.norm(all_doc_embeddings, axis=1)
    
    valid = doc_norms > 0
    distances = np.ones(len(doc_norms))
    distances[valid] = 1 - (dot_products[valid] / (doc_norms[valid] * query_norm))
    
    # Get target distance
    target_distance = cosine_distance(query_embedding, doc_embedding)
    
    # Count how many docs are closer
    rank = np.sum(distances < target_distance) + 1
    
    return rank, target_distance, distances

def main():
    """Interactive search and ranking tool."""
    
    print("Document Ranking Tool")
    print("=" * 80)
    
    # Load vector database
    print("\nLoading vector database...")
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"Error: Vector database not found at {VECTOR_DB_PATH}")
        return
    
    with open(VECTOR_DB_PATH, 'rb') as f:
        vector_db = pickle.load(f)
    
    doc_names_list = sorted(list(vector_db.keys()))
    doc_embeddings_array = np.array([np.array(vector_db[name]['embedding'], dtype=np.float32) 
                                     for name in doc_names_list], dtype=np.float32)
    
    print(f"✓ Loaded {len(vector_db)} document embeddings\n")
    
    # Interactive loop
    while True:
        print("=" * 80)
        print("\nOptions:")
        print("  1. Search for a document by query and path")
        print("  2. Exit")
        
        choice = input("\nSelect option (1 or 2): ").strip()
        
        if choice == '2':
            print("Goodbye!")
            break
        
        if choice != '1':
            print("Invalid choice. Please select 1 or 2.")
            continue
        
        # Get search query
        query_text = input("\nEnter search query: ").strip()
        if not query_text:
            print("Query cannot be empty.")
            continue
        
        # Get document path
        doc_path = input("Enter document path: ").strip()
        if not doc_path:
            print("Document path cannot be empty.")
            continue
        
        # Get embeddings
        print("\nProcessing...")
        query_embedding = get_query_embedding(query_text)
        if query_embedding is None:
            continue
        
        doc_embedding, doc_content = get_document_embedding(doc_path)
        if doc_embedding is None:
            continue
        
        # Rank document
        rank, target_distance, all_distances = rank_document(query_embedding, doc_embedding, doc_embeddings_array)
        target_similarity = 1 - target_distance
        
        # Get top-10 documents
        sorted_indices = np.argsort(all_distances)
        top_10_docs = [doc_names_list[i] for i in sorted_indices[:10]]
        top_10_sims = [1 - all_distances[i] for i in sorted_indices[:10]]
        
        # Display results
        print("\n" + "=" * 80)
        print("RANKING RESULTS")
        print("=" * 80)
        
        doc_name = os.path.basename(doc_path)
        print(f"\nQuery: {query_text}")
        print(f"Document: {doc_name}")
        print(f"\nRanking Position: #{rank} out of {len(vector_db)} documents")
        print(f"Similarity Score: {target_similarity:.4f}")
        
        print(f"\nTop-10 Retrieved Documents:")
        for rank_pos, (doc, sim) in enumerate(zip(top_10_docs, top_10_sims), 1):
            is_target = " <-- YOUR DOCUMENT" if doc == doc_name else ""
            print(f"  {rank_pos:2d}. {doc[:70]:70s} (sim: {sim:.4f}){is_target}")
        
        # Show where target document appears if beyond top-10
        if rank > 10:
            print(f"\n  ... ({rank - 10} documents ranked higher than yours)")
        
        print()

if __name__ == '__main__':
    main()
