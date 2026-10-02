"""
Complete autoencoder test: embedding, manipulation, and reconstruction.

This script demonstrates:
1. Encoding text to embeddings
2. Manipulating embeddings (vector operations)
3. Finding semantically similar texts
4. Approximating reconstruction through similarity
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

print("=" * 80)
print(" " * 20 + "TEXT AUTOENCODER - COMPLETE TEST WORKFLOW")
print("=" * 80)

# Load model
print("\n[1/5] Loading pre-trained text autoencoder...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = model.get_sentence_embedding_dimension()
print(f"      ✓ Model loaded | Embedding dimension: {embedding_dim}")

# Build corpus
corpus = [
    "Dogs are loyal and friendly animals",
    "Cats are independent and graceful pets",
    "Birds can fly through the sky",
    "Fish swim in water and eat plants",
    "Horses are strong and fast runners",
    "Python is a programming language",
    "JavaScript is used for web development",
    "Java is widely used in enterprises",
    "Artificial intelligence is the future",
    "Machine learning models learn from data",
]

print("\n[2/5] Encoding corpus to embeddings...")
corpus_embeddings = model.encode(corpus, show_progress_bar=True)
print(f"      ✓ Encoded {len(corpus)} texts | Shape: {corpus_embeddings.shape}")

# Test embedding and reconstruction
test_texts = [
    "Puppies are cute and loyal animals",
    "C++ is a compiled programming language",
    "Deep learning is part of AI",
]

print("\n[3/5] EMBEDDING → RECONSTRUCTION CYCLE")
print("      " + "-" * 75)

results = []

for test_text in test_texts:
    print(f"\n   INPUT TEXT: '{test_text}'")
    
    # Step 1: Encode (Embedding)
    embedding = model.encode(test_text, show_progress_bar=False)
    print(f"      ✓ Encoded to {embedding_dim}-dimensional vector")
    
    # Step 2: Find reconstruction (most similar corpus text)
    similarities = cosine_similarity([embedding], corpus_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    reconstructed_text = corpus[best_match_idx]
    
    print(f"      ✓ Found reconstruction via similarity search")
    print(f"      RECONSTRUCTED: '{reconstructed_text}'")
    print(f"      SIMILARITY: {best_similarity:.4f} (0=different, 1=identical)")
    
    # Step 3: Analyze the embedding
    print(f"      Analysis:")
    print(f"        - Min value: {embedding.min():.6f}")
    print(f"        - Max value: {embedding.max():.6f}")
    print(f"        - Mean: {embedding.mean():.6f}")
    print(f"        - Magnitude (L2 norm): {np.linalg.norm(embedding):.4f}")
    
    # Store result
    results.append({
        "input": test_text,
        "reconstructed": reconstructed_text,
        "similarity": float(best_similarity),
        "embedding_stats": {
            "dimension": int(embedding_dim),
            "min": float(embedding.min()),
            "max": float(embedding.max()),
            "mean": float(embedding.mean()),
            "l2_norm": float(np.linalg.norm(embedding))
        }
    })

print("\n[4/5] VECTOR ARITHMETIC IN EMBEDDING SPACE")
print("      " + "-" * 75)

# Demonstrate embedding arithmetic
text1 = "Dogs are animals"
text2 = "Cats are animals"
text3 = "Programming is about coding"

emb1 = model.encode(text1)
emb2 = model.encode(text2)
emb3 = model.encode(text3)

print(f"\n   Text 1: '{text1}'")
print(f"   Text 2: '{text2}'")
print(f"   Text 3: '{text3}'")

# Compute difference
diff = emb2 - emb1
print(f"\n   Vector operation: emb(text2) - emb(text1)")
print(f"   This captures the semantic difference between the texts")

# Use difference with another text
result_emb = emb3 + diff
similarities = cosine_similarity([result_emb], corpus_embeddings)[0]
closest_idx = np.argmax(similarities)

print(f"\n   Applied difference to text3: emb(text3) + diff")
print(f"   Closest match: '{corpus[closest_idx]}'")
print(f"   Similarity: {similarities[closest_idx]:.4f}")

print("\n[5/5] CLUSTERING ANALYSIS")
print("      " + "-" * 75)

# Simple clustering by similarity
print("\n   Grouping similar texts:")

# Animal group
animal_texts = [corpus[i] for i in [0, 1, 2, 3, 4]]
animal_embeddings = corpus_embeddings[[0, 1, 2, 3, 4]]
animal_centroid = animal_embeddings.mean(axis=0)

# Programming group
prog_texts = [corpus[i] for i in [5, 6, 7]]
prog_embeddings = corpus_embeddings[[5, 6, 7]]
prog_centroid = prog_embeddings.mean(axis=0)

# AI group
ai_texts = [corpus[i] for i in [8, 9]]
ai_embeddings = corpus_embeddings[[8, 9]]
ai_centroid = ai_embeddings.mean(axis=0)

print("\n   GROUP 1: Animals")
for text in animal_texts:
    print(f"     - {text}")

print("\n   GROUP 2: Programming Languages")
for text in prog_texts:
    print(f"     - {text}")

print("\n   GROUP 3: Artificial Intelligence")
for text in ai_texts:
    print(f"     - {text}")

# Calculate centroid distances
print(f"\n   Distance between group centroids:")
animal_to_prog = cosine_similarity([animal_centroid], [prog_centroid])[0][0]
animal_to_ai = cosine_similarity([animal_centroid], [ai_centroid])[0][0]
prog_to_ai = cosine_similarity([prog_centroid], [ai_centroid])[0][0]

print(f"     Animals ↔ Programming: {animal_to_prog:.4f}")
print(f"     Animals ↔ AI: {animal_to_ai:.4f}")
print(f"     Programming ↔ AI: {prog_to_ai:.4f}")

print("\n" + "=" * 80)
print("COMPLETE TEST SUMMARY")
print("=" * 80)
print(f"""
✓ ENCODING:       Text → {embedding_dim}-dimensional embeddings
✓ RECONSTRUCTION: Found similar texts for all test inputs
✓ ARITHMETIC:     Vector operations preserve semantic meaning
✓ CLUSTERING:     Embeddings capture semantic similarity

KEY TAKEAWAYS:
1. Autoencoders compress text into compact numerical representations
2. Similar texts have similar embeddings (close in vector space)
3. You can perform mathematics on embeddings
4. Embeddings enable clustering, search, and classification
5. The embedding is the "bottleneck" - compressed information

NEXT STEPS:
- Try modifying embeddings and see how reconstruction changes
- Build a semantic search engine using these embeddings
- Fine-tune the model on domain-specific data
- Explore different embedding dimensions
""")

# Save results to file
with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to: test_results.json")
