"""
Learn about the DECODER part of autoencoders.

While sentence-transformers primarily focus on the ENCODER (text → embedding),
real autoencoders also have a DECODER (embedding → reconstructed text).

This script explains the decoder concept and shows approximate reconstruction
using similarity search (a practical alternative).
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("TEXT AUTOENCODER - DECODING (RECONSTRUCTION) TUTORIAL")
print("=" * 70)

# Load the model
print("\n1. Loading the pre-trained model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"   ✓ Model loaded!")

# Create a small "corpus" of known texts
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A fast orange fox leaps above a slow dog",
    "The slow brown cat naps in the sun",
    "Machine learning is powerful",
    "Deep learning uses neural networks",
    "Artificial intelligence is transforming society",
    "The sun is shining brightly",
    "It is a beautiful sunny day",
    "Weather is cloudy and cold today",
]

print("\n2. UNDERSTANDING THE DECODER")
print("   " + "-" * 65)
print("""
In a TRUE autoencoder:
  ┌─────────┐      ┌─────────┐      ┌──────────┐
  │  Text   │──→   │Embedding│  ──→ │Reconstructed│
  │  INPUT  │      │(Bottleneck)    │    TEXT     │
  └─────────┘      └─────────┘      └──────────┘
    ENCODER              CODE           DECODER

The DECODER learns to reconstruct from the compressed embedding.

However, sentence-transformers is primarily an ENCODER-only model.
To approximate decoding, we can:
  1. Take the embedding of new text
  2. Find the most similar text in our corpus
  3. Return that as the "decoded" result
""")

print("\n3. ENCODING: Convert corpus texts to embeddings...")
corpus_embeddings = model.encode(corpus, show_progress_bar=False)
print(f"   ✓ Encoded {len(corpus)} texts")
print(f"   Each embedding shape: {corpus_embeddings.shape[1]}")

print("\n4. APPROXIMATE DECODING via Similarity Search")
print("   " + "-" * 65)

# Test texts (different from corpus)
test_texts = [
    "A speedy crimson fox jumps over a lazy dog",  # Similar to fox sentence
    "Neural networks are great for learning",      # Similar to ML sentences
    "Beautiful sunny weather today",               # Similar to weather sentences
]

print("\nFor each test text:")
print("  1. Encode it to embedding")
print("  2. Find most similar corpus text")
print("  3. Return corpus text as 'decoded' result\n")

for test_text in test_texts:
    print(f"INPUT (NEW TEXT): '{test_text}'")
    
    # Encode the test text
    test_embedding = model.encode(test_text, show_progress_bar=False)
    
    # Find most similar text in corpus
    similarities = cosine_similarity([test_embedding], corpus_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[most_similar_idx]
    
    print(f"DECODED OUTPUT:  '{corpus[most_similar_idx]}'")
    print(f"SIMILARITY:       {similarity_score:.4f}")
    
    # Show top 3 matches
    print(f"Top 3 matches:")
    top_indices = np.argsort(similarities)[-3:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. ({similarities[idx]:.4f}) {corpus[idx]}")
    print()

print("=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)
print("""
1. REAL AUTOENCODER DECODER:
   - Trained on the same data as encoder
   - Learns inverse transformation (embedding → text)
   - Progressively reconstructs original text
   - Reconstruction loss trains the whole network

2. OUR APPROXIMATION:
   - Uses similarity search as "decoding"
   - Practical for retrieval tasks
   - Shows that embeddings preserve semantic info
   - Loss would be: distance between original & reconstructed

3. AUTOENCODER BENEFITS:
   - Dimensionality reduction (384 dims → compressed)
   - Feature learning (meaningful representations)
   - Anomaly detection
   - Data denoising
   - Can be used for clustering & search
""")

print("\n✓ Understanding autoencoders requires thinking in embedding space!")
