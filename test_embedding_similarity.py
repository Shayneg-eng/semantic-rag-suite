import ollama
import numpy as np
from scipy.spatial.distance import cosine

EMBEDDING_MODEL = "nomic-embed-text"

def get_embedding(text):
    """Get embedding from ollama."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        if "embeddings" in response and len(response["embeddings"]) > 0:
            return np.array(response["embeddings"][0])
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0
    return 1 - cosine(vec1, vec2)

def main():
    print("=" * 70)
    print("EMBEDDING SIMILARITY TEST")
    print("=" * 70)
    print("\nType words/phrases to test embeddings.")
    print("Type 'quit' to exit.\n")
    
    last_embedding = None
    last_text = None
    
    while True:
        text = input("Enter a word or phrase: ").strip()
        
        if text.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter something.\n")
            continue
        
        print(f"\nEmbedding '{text}'...")
        embedding = get_embedding(text)
        
        if embedding is None:
            print("Failed to get embedding.\n")
            continue
        
        print(f"✓ Got embedding ({len(embedding)} dimensions)")
        print(f"  First 5 dims: {embedding[:5]}")
        print(f"  Non-zero values: {np.count_nonzero(embedding)}")
        
        if last_embedding is not None:
            similarity = cosine_similarity(embedding, last_embedding)
            print(f"\nSimilarity to '{last_text}': {similarity:.4f}")
            print(f"  (0 = opposite, 1 = identical)")
        
        last_embedding = embedding
        last_text = text
        print()

if __name__ == "__main__":
    main()
