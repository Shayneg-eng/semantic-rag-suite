import csv
import numpy as np
from ollama import embed
import os

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

def search(query, top_k=5):
    """Search for most similar documents"""
    print(f"\nSearching for: '{query}'")
    print("Embedding query...")
    
    query_embedding = get_embedding(query)
    
    # Calculate similarity to all documents
    similarities = []
    for doc_name, doc_embedding in doc_embeddings.items():
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_name, sim, doc_paths[doc_name]))
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_k} results:")
    print("-" * 80)
    for i, (doc_name, sim, doc_path) in enumerate(similarities[:top_k], 1):
        print(f"{i}. {doc_name}")
        print(f"   Similarity: {sim:.4f}")
        print(f"   Path: {doc_path}")
        print()

# Interactive search loop
while True:
    query = input("Enter search query (or 'quit' to exit): ").strip()
    if query.lower() == 'quit':
        break
    if query:
        search(query)
