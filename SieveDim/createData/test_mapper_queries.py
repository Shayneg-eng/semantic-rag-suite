import os
import csv
import pickle
import numpy as np
import torch
import random
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'query_document_pairs.csv')
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'vector_db.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'query_embedding_mapper.pt')

EMBEDDING_DIM = 768
HIDDEN_DIM = 512

# Load model code
import sys
sys.path.append(os.path.dirname(__file__))
from train_embedding_mapper import EmbeddingMapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """Test 10 random queries using the trained embedding mapper."""
    
    print("Testing Embedding Mapper on Random Queries")
    print("=" * 100)
    
    # Load model
    print("\nLoading trained model...")
    model = EmbeddingMapper(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
    
    # Load vector database
    print("Loading vector database...")
    with open(VECTOR_DB_PATH, 'rb') as f:
        vector_db = pickle.load(f)
    print(f"✅ Loaded {len(vector_db)} document embeddings")
    
    # Prepare for vectorized search
    doc_names_list = sorted(list(vector_db.keys()))
    doc_embeddings_array = np.array([np.array(vector_db[name]['embedding'], dtype=np.float32) 
                                     for name in doc_names_list], dtype=np.float32)
    print(f"✅ Prepared embeddings for fast retrieval\n")
    
    # Load CSV
    print("Loading query-document pairs...")
    queries = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    print(f"✅ Loaded {len(queries)} query-document pairs")
    
    # Select 10 random queries
    random_indices = random.sample(range(len(queries)), min(10, len(queries)))
    random_queries = [queries[i] for i in random_indices]
    
    print(f"\n{'='*100}")
    print(f"Testing {len(random_queries)} random queries with Embedding Mapper")
    print(f"{'='*100}\n")
    
    for test_num, row in enumerate(random_queries, 1):
        query_text = row['query']
        correct_doc = row['correct_document']
        query_embedding = eval(row['query_embedding'])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Get mapped embedding using the trained model
        with torch.no_grad():
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(device)
            mapped_query_embedding = model(query_tensor).squeeze(0).cpu().numpy()
        
        # Compute distances using MAPPED embedding
        mapped_query_embedding = np.array(mapped_query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(mapped_query_embedding)
        
        dot_products = doc_embeddings_array @ mapped_query_embedding
        doc_norms = np.linalg.norm(doc_embeddings_array, axis=1)
        
        valid = doc_norms > 0
        distances = np.ones(len(doc_norms))
        distances[valid] = 1 - (dot_products[valid] / (doc_norms[valid] * query_norm))
        
        sorted_indices = np.argsort(distances)
        sorted_docs = [doc_names_list[i] for i in sorted_indices]
        sorted_distances = [distances[i] for i in sorted_indices]
        
        # Find position of correct document
        try:
            position = sorted_docs.index(correct_doc) + 1
        except ValueError:
            position = None
        
        # Display results
        print(f"Query #{test_num}")
        print(f"{'─' * 100}")
        print(f"Query Text: {query_text}")
        print(f"Correct Document: {correct_doc}")
        if position:
            print(f"✓ Position Ranked: #{position} out of {len(vector_db)} documents")
        else:
            print(f"✗ Document not found in vector database")
        
        print(f"\nTop-10 Retrieved Documents:")
        for rank, (doc, dist) in enumerate(zip(sorted_docs[:10], sorted_distances[:10]), 1):
            is_correct = "✓ CORRECT" if doc == correct_doc else ""
            sim = 1 - dist  # Convert distance back to similarity
            print(f"  {rank:2d}. {doc[:70]:70s} (sim: {sim:7.4f}) {is_correct}")
        
        if position and position > 10:
            print(f"\n  ... {position - 10} documents ranked before correct document ...")
        
        print(f"\n")
    
    print("=" * 100)

if __name__ == '__main__':
    main()
