"""
Learn how to embed text using a text autoencoder.

An autoencoder encodes high-dimensional data (text) into a lower-dimensional
representation (embedding) and can decode it back.

This script demonstrates the ENCODING step.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

print("=" * 70)
print("TEXT AUTOENCODER - EMBEDDING (ENCODING) TUTORIAL")
print("=" * 70)

# Load the pre-trained model
print("\n1. Loading the pre-trained sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"   ✓ Model loaded!")
print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Example texts to embed
example_texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Text autoencoders can compress and reconstruct text",
    "The weather is sunny today",
    "I love to read books and learn new things"
]

print("\n2. EMBEDDING (ENCODING) PROCESS")
print("   Converting text → numerical vectors (embeddings)")
print("   " + "-" * 65)

# Encode the texts
embeddings = model.encode(example_texts, show_progress_bar=False)

print(f"\n   Input: {len(example_texts)} text samples")
print(f"   Output: {embeddings.shape[0]} embeddings")
print(f"   Each embedding dimension: {embeddings.shape[1]}")

# Display detailed information
for i, (text, embedding) in enumerate(zip(example_texts, embeddings)):
    print(f"\n   [{i+1}] Original Text:")
    print(f"       {text}")
    print(f"       Embedding (first 10 values): {embedding[:10]}")
    print(f"       Embedding length: {len(embedding)}")
    print(f"       Mean value: {embedding.mean():.4f}")
    print(f"       Std dev: {embedding.std():.4f}")

print("\n" + "=" * 70)
print("KEY CONCEPTS:")
print("=" * 70)
print("""
1. ENCODER: The model compresses text into a fixed-size vector (embedding)
   - Input: Variable-length text
   - Output: Fixed-size numerical vector (e.g., 384 dimensions)

2. EMBEDDING: The compressed representation
   - Captures semantic meaning
   - Similar texts have similar embeddings (closer in vector space)
   - Can be used for similarity search, clustering, classification

3. WHY IT WORKS:
   - The encoder learns semantic relationships during training
   - Words with similar meanings get similar embeddings
   - The embedding space is "meaningful" - nearby points = similar meanings
""")

# Demonstrate similarity
print("\n" + "=" * 70)
print("BONUS: SIMILARITY SEARCH")
print("=" * 70)

from sklearn.metrics.pairwise import cosine_similarity

print("\nComparing embeddings using cosine similarity (0-1 scale):")
print(f"  1.0 = identical meaning, 0.0 = completely different")

# Compare first text with all others
similarities = cosine_similarity([embeddings[0]], embeddings)[0]

print(f"\nText 1: '{example_texts[0]}'")
print("\nSimilarity to all texts:")
for i, sim in enumerate(similarities):
    print(f"  [{i+1}] {sim:.4f} - {example_texts[i]}")

print("\n✓ The autoencoder successfully created meaningful embeddings!")
