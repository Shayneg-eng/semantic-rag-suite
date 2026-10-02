"""
Download a pre-trained text autoencoder model.

We'll use a Variational Autoencoder (VAE) through sentence-transformers
and also explore huggingface transformers for a more direct autoencoder example.
"""

import os
from pathlib import Path

# Create a models directory
models_dir = Path("./models")
models_dir.mkdir(exist_ok=True)

print("=" * 60)
print("TEXT AUTOENCODER - MODEL DOWNLOAD")
print("=" * 60)

print("\n1. Installing required libraries...")
try:
    import sentence_transformers
    print("   ✓ sentence-transformers already installed")
except ImportError:
    print("   Installing sentence-transformers...")
    os.system("pip install sentence-transformers")

try:
    import transformers
    print("   ✓ transformers already installed")
except ImportError:
    print("   Installing transformers...")
    os.system("pip install transformers")

try:
    import torch
    print("   ✓ torch already installed")
except ImportError:
    print("   Installing torch...")
    os.system("pip install torch")

print("\n2. Downloading models...")
print("\n   Model 1: Sentence Transformers (for embeddings)")
print("   - All-MiniLM-L6-v2 (lightweight, good for learning)")

from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
print(f"\n   Downloading '{model_name}'...")
model = SentenceTransformer(model_name)
print(f"   ✓ Model downloaded and cached!")
print(f"   Model path: {model_name}")
print(f"   Output dimension: {model.get_sentence_embedding_dimension()}")

print("\n" + "=" * 60)
print("SETUP COMPLETE!")
print("=" * 60)
print("\nYou can now use the model in other scripts.")
print("The model has been cached locally.")
print("\nNext steps:")
print("  - Run: python 2_embed_text.py")
print("  - Run: python 3_unembed_text.py")
print("  - Run: python 4_autoencoder_test.py")
