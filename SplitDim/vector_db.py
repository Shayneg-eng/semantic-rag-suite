import os
import json
import ollama
from pathlib import Path
from typing import List, Tuple
import numpy as np

class VectorDatabase:
    def __init__(self, docs_path: str, model: str = 'nomic-embed-text'):
        self.docs_path = docs_path
        self.model = model
        self.documents = {}  # {doc_name: content}
        self.embeddings = {}  # {doc_name: embedding}
        self.db_file = 'vector_db.json'
        
    def load_documents(self):
        """Load all text documents from the directory"""
        print(f"Loading documents from {self.docs_path}...")
        docs_dir = Path(self.docs_path)
        
        for file_path in sorted(docs_dir.glob('*.txt')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc_name = file_path.name
                    self.documents[doc_name] = content
                    print(f"  Loaded: {doc_name}")
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
        
        print(f"Total documents loaded: {len(self.documents)}\n")
    
    def create_embeddings(self):
        """Create embeddings for all documents"""
        print(f"Creating embeddings using model '{self.model}'...")
        
        for doc_name, content in self.documents.items():
            try:
                # Use the beginning of the document as the embedding input
                # (you could also summarize or use the full content)
                text_to_embed = content[:500] if len(content) > 500 else content
                
                response = ollama.embed(
                    model=self.model,
                    input=text_to_embed,
                )
                
                embedding = response['embeddings'][0] if response['embeddings'] else None
                if embedding:
                    self.embeddings[doc_name] = embedding
                    print(f"  ✓ {doc_name}")
                else:
                    print(f"  ✗ {doc_name} (no embedding returned)")
                    
            except Exception as e:
                print(f"  ✗ {doc_name}: {e}")
        
        print(f"Embeddings created: {len(self.embeddings)}\n")
    
    def save_database(self):
        """Save embeddings to file"""
        data = {
            'model': self.model,
            'documents': self.documents,
            'embeddings': self.embeddings
        }
        
        # Convert numpy arrays to lists for JSON serialization
        embeddings_serializable = {
            doc_name: embedding if isinstance(embedding, list) else list(embedding)
            for doc_name, embedding in self.embeddings.items()
        }
        data['embeddings'] = embeddings_serializable
        
        with open(self.db_file, 'w') as f:
            json.dump(data, f)
        print(f"Database saved to {self.db_file}\n")
    
    def load_database(self):
        """Load embeddings from file"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
            print(f"Database loaded from {self.db_file}")
            print(f"Documents: {len(self.documents)}, Embeddings: {len(self.embeddings)}\n")
            return True
        return False
    
    def cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return float(dot_product / (magnitude1 * magnitude2))
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        try:
            # Create embedding for query
            response = ollama.embed(
                model=self.model,
                input=query,
            )
            query_embedding = response['embeddings'][0]
            
            # Calculate similarity scores
            scores = {}
            for doc_name, embedding in self.embeddings.items():
                similarity = self.cosine_similarity(query_embedding, embedding)
                scores[doc_name] = similarity
            
            # Sort by similarity score (descending)
            sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_results[:top_k]
        
        except Exception as e:
            print(f"Error during search: {e}")
            return []


def main():
    # Create and build the vector database
    db = VectorDatabase('legal_rag_test_documents')
    
    # Try to load existing database, otherwise create new one
    if not db.load_database():
        db.load_documents()
        db.create_embeddings()
        db.save_database()
    
    # Interactive search
    print("=" * 70)
    print("VECTOR DATABASE SEARCH")
    print("=" * 70)
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter search query: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if not query:
            print("Please enter a query\n")
            continue
        
        print(f"\nSearching for: '{query}'")
        print("-" * 70)
        
        results = db.search(query, top_k=5)
        
        if results:
            for i, (doc_name, score) in enumerate(results, 1):
                print(f"\n{i}. {doc_name}")
                print(f"   Similarity Score: {score:.4f}")
                # Print first 200 chars of the document
                preview = db.documents[doc_name][:200].replace('\n', ' ')
                print(f"   Preview: {preview}...")
        else:
            print("No results found")
        
        print("-" * 70 + "\n")


if __name__ == '__main__':
    main()
