import os
import json
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
WINDOW_SIZE = 51  # Sliding window size (changeable)
EMBEDDING_MODEL = "nomic-embed-text"
NUM_WORKERS = 10  # Number of parallel workers
BATCH_SIZE = 50  # Batch multiple embeddings in one API call
MAX_CONTEXT_CHARS = 2000  # Max characters per embedding (roughly 2K tokens)

CORPUS_DIR = Path("LegalBench-RAG/corpus_flat")
EMBEDDINGS_DIR = Path("embeddings")
MASTER_CSV = EMBEDDINGS_DIR / "master_embeddings.csv"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_ollama_running():
    """Check if Ollama is running, if not start it."""
    try:
        # Try to pull a simple model to check if ollama is running
        ollama.pull(EMBEDDING_MODEL)
        print("✓ Ollama is running")
        return True
    except:
        print("Starting Ollama...")
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        import time
        time.sleep(3)
        return True

def pull_model():
    """Ensure the embedding model is pulled."""
    print(f"Pulling model {EMBEDDING_MODEL}...")
    try:
        ollama.pull(EMBEDDING_MODEL)
        print(f"✓ Model {EMBEDDING_MODEL} ready")
        return True
    except Exception as e:
        print(f"✗ Failed to pull model: {e}")
        return False

def get_embedding(text):
    """Get embedding from ollama using the Python library."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        if "embeddings" in response and len(response["embeddings"]) > 0:
            return response["embeddings"][0]
        return None
    except Exception as e:
        # Silently fail on context length errors and return None
        if "context length" not in str(e).lower():
            print(f"Error getting embedding: {e}")
        return None

def get_batch_embeddings(texts):
    """Get embeddings for multiple texts in one call (faster)."""
    if not texts:
        return []
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
        if "embeddings" in response:
            return response["embeddings"]
        return [None] * len(texts)
    except Exception as e:
        # Return None for all if context length exceeded
        if "context length" not in str(e).lower():
            print(f"Error getting batch embeddings: {e}")
        return [None] * len(texts)

def split_text_into_chunks(text, max_chars=MAX_CONTEXT_CHARS):
    """Split text into chunks of max_chars or less."""
    if len(text) <= max_chars:
        return [text]
    
    # Split on word boundaries
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len <= max_chars:
            current_chunk.append(word)
            current_length += word_len
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embedding_chunked(text):
    """Get embedding by chunking if needed and averaging."""
    if len(text) <= MAX_CONTEXT_CHARS:
        return get_embedding(text)
    
    # Split into chunks
    chunks = split_text_into_chunks(text, MAX_CONTEXT_CHARS)
    
    # Get embeddings for each chunk
    chunk_embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        if embedding is not None:
            chunk_embeddings.append(embedding)
    
    if not chunk_embeddings:
        return None
    
    # Average the embeddings
    return np.mean(chunk_embeddings, axis=0).tolist()

def tokenize_words(text):
    """Split text into words, preserving word boundaries."""
    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b|\S', text)
    return words

def get_sliding_window_embeddings(words):
    """
    Generate sliding window embeddings with chunking support.
    
    Returns: list of (word_index, word, embedding) tuples
    """
    half_window = WINDOW_SIZE // 2
    results = []
    
    # Process every word
    indices_to_process = list(range(len(words)))
    print(f"    Generating embeddings for {len(indices_to_process)} words (window={WINDOW_SIZE})...")
    
    for i in tqdm(indices_to_process, desc="    Processing", leave=False):
        # Define window boundaries
        start = max(0, i - half_window)
        end = min(len(words), i + half_window + 1)
        
        # Get words in window
        window_words = words[start:end]
        window_text = " ".join(window_words)
        
        # Get embedding (with chunking if needed)
        embedding = get_embedding_chunked(window_text)
        
        if embedding is not None:
            results.append((i, words[i], embedding))
        else:
            results.append((i, words[i], [0.0] * 384))
    
    return results

def process_document(doc_path):
    """Process a single document and generate embeddings."""
    doc_name = doc_path.stem
    start_time = time.time()
    
    try:
        # Read document with fallback encoding handling
        try:
            with open(doc_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try latin-1 as fallback
            with open(doc_path, 'r', encoding='latin-1', errors='replace') as f:
                text = f.read()
        
        # Tokenize
        words = tokenize_words(text)
        
        if len(words) == 0:
            return None
        
        # Get full document embedding (with chunking if needed)
        doc_embedding = get_embedding_chunked(text[:5000])  # Use first 5000 chars for context
        
        # Get sliding window embeddings
        word_embeddings = get_sliding_window_embeddings(words)
        
        # Create document folder
        doc_folder = EMBEDDINGS_DIR / doc_name
        doc_folder.mkdir(parents=True, exist_ok=True)
        
        # Save word-level embeddings to CSV
        word_csv_path = doc_folder / f"{doc_name}_word_embeddings.csv"
        
        with open(word_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            header = ['word_index', 'word'] + [f'dim_{i}' for i in range(len(word_embeddings[0][2]))]
            writer.writerow(header)
            # Write data
            for word_idx, word, embedding in word_embeddings:
                row = [word_idx, word] + embedding
                writer.writerow(row)
        
        elapsed = time.time() - start_time
        
        return {
            'document': doc_name,
            'word_count': len(words),
            'embedding_count': len(word_embeddings),
            'embedding_dim': len(word_embeddings[0][2]),
            'doc_embedding': doc_embedding if doc_embedding else [0.0] * 384,
            'csv_path': str(word_csv_path),
            'elapsed_time': elapsed
        }
        
    except Exception as e:
        return None

def process_all_documents():
    """Process all documents in the corpus using parallel processing."""
    if not CORPUS_DIR.exists():
        print(f"✗ Corpus directory not found: {CORPUS_DIR}")
        return False
    
    # Get all text files
    doc_files = list(CORPUS_DIR.glob("*.txt"))
    print(f"\n📂 Found {len(doc_files)} documents to process")
    
    if len(doc_files) == 0:
        print("✗ No text files found in corpus")
        return False
    
    # Process documents in parallel
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    master_data = []
    
    print(f"⚙️  Using {NUM_WORKERS} parallel workers")
    print(f"📊 Batch size: {BATCH_SIZE} | Context limit: {MAX_CONTEXT_CHARS} chars")
    print("-" * 70)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_doc = {executor.submit(process_document, doc_path): doc_path for doc_path in doc_files}
        
        # Collect results as they complete
        with tqdm(total=len(doc_files), desc="Processing documents", unit="doc", 
                  bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for future in as_completed(future_to_doc):
                result = future.result()
                if result:
                    master_data.append(result)
                    pbar.set_postfix({
                        'processed': len(master_data),
                        'avg_time': f"{sum(r.get('elapsed_time', 0) for r in master_data) / len(master_data):.1f}s"
                    })
                pbar.update(1)
    
    elapsed_total = time.time() - start_time
    
    # Save master CSV
    if master_data:
        print(f"\n💾 Saving master embeddings CSV...")
        with open(MASTER_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            header = ['document', 'word_count', 'embedding_count', 'embedding_dim', 'csv_path'] + \
                    [f'doc_embedding_dim_{i}' for i in range(len(master_data[0]['doc_embedding']))]
            writer.writerow(header)
            # Write data
            for item in master_data:
                row = [item['document'], item['word_count'], item['embedding_count'], 
                      item['embedding_dim'], item['csv_path']] + item['doc_embedding']
                writer.writerow(row)
        
        print(f"✅ Saved master embeddings to {MASTER_CSV}")
        print(f"\n📈 Processing Summary:")
        print(f"  • Total documents: {len(master_data)}")
        print(f"  • Total words embedded: {sum(r['word_count'] for r in master_data):,}")
        print(f"  • Total embeddings: {sum(r['embedding_count'] for r in master_data):,}")
        print(f"  • Total time: {elapsed_total/60:.1f} minutes")
        print(f"  • Avg time per doc: {elapsed_total/len(master_data):.1f}s")
        return True
    else:
        print("✗ No documents were processed successfully")
        return False

def main():
    print("\n" + "=" * 70)
    print("🚀 LEGAL DOCUMENT EMBEDDING GENERATOR")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"  • Model: {EMBEDDING_MODEL}")
    print(f"  • Window Size: {WINDOW_SIZE} words")
    print(f"  • Max Context: {MAX_CONTEXT_CHARS} characters")
    print(f"  • Batch Size: {BATCH_SIZE}")
    print(f"  • Workers: {NUM_WORKERS}")
    print(f"  • Corpus: {CORPUS_DIR}")
    print(f"  • Output: {EMBEDDINGS_DIR}")
    print("=" * 70)
    
    # Step 1: Ensure Ollama is running
    print("\n[1/4] 🔍 Checking Ollama...")
    if not ensure_ollama_running():
        print("✗ Failed to start Ollama")
        return False
    
    # Step 2: Pull model
    print("\n[2/4] 📦 Pulling embedding model...")
    if not pull_model():
        print("✗ Failed to pull model")
        return False
    
    # Step 3: Process documents
    print("\n[3/4] 🔄 Processing documents...")
    if not process_all_documents():
        print("✗ Failed to process documents")
        return False
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("✨ SUCCESS! Embeddings generated.")
    print("=" * 70)
    print(f"\n📁 Output structure:")
    print(f"  embeddings/")
    print(f"    ├── master_embeddings.csv")
    print(f"    ├── document1/")
    print(f"    │   └── document1_word_embeddings.csv")
    print(f"    ├── document2/")
    print(f"    │   └── document2_word_embeddings.csv")
    print(f"    └── ...")
    print(f"\n📊 CSV Format:")
    print(f"  • Column 1: word_index (position in document)")
    print(f"  • Column 2: word")
    print(f"  • Columns 3+: embedding dimensions (384 dims for {EMBEDDING_MODEL})")
    print("=" * 70 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
