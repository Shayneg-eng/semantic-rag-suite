import os
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import ollama
from scipy.spatial.distance import cosine

# ============================================================================
# CONFIGURATION
# ============================================================================
EMBEDDING_MODEL = "nomic-embed-text"
MAX_CONTEXT_CHARS = 2000  # Max characters per embedding

CORPUS_DIR = Path("LegalBench-RAG/corpus_flat")
EMBEDDINGS_DIR = Path("embeddings")
DOCUMENT_EMBEDDINGS_CSV = EMBEDDINGS_DIR / "document_embeddings.csv"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding(text):
    """Get embedding from ollama."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        if "embeddings" in response and len(response["embeddings"]) > 0:
            return np.array(response["embeddings"][0])
        return None
    except Exception as e:
        if "context length" not in str(e).lower():
            print(f"Error: {e}")
        return None

def split_text_into_chunks(text, max_chars=MAX_CONTEXT_CHARS):
    """Split text into chunks of max_chars or less."""
    if len(text) <= max_chars:
        return [text]
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1
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

def get_document_embedding(text):
    """Get embedding for entire document, with chunking if needed."""
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
    return np.mean(chunk_embeddings, axis=0)

def main():
    print("\n" + "=" * 70)
    print("📄 DOCUMENT EMBEDDING GENERATOR (Fast Version)")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"  • Model: {EMBEDDING_MODEL}")
    print(f"  • Max Context: {MAX_CONTEXT_CHARS} characters")
    print(f"  • Corpus: {CORPUS_DIR}")
    print(f"  • Output: {DOCUMENT_EMBEDDINGS_CSV}")
    print("=" * 70)
    
    # Get all documents
    if not CORPUS_DIR.exists():
        print(f"✗ Corpus not found: {CORPUS_DIR}")
        return False
    
    doc_files = sorted(list(CORPUS_DIR.glob("*.txt")))
    print(f"\n📂 Found {len(doc_files)} documents")
    
    if len(doc_files) == 0:
        print("✗ No documents found")
        return False
    
    # Create embeddings directory
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process documents
    print(f"\n🔄 Embedding documents...\n")
    start_time = time.time()
    
    embeddings_data = []
    
    with tqdm(total=len(doc_files), desc="Processing", unit="doc") as pbar:
        for doc_path in doc_files:
            try:
                # Read document
                with open(doc_path, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                
                if len(text.strip()) == 0:
                    pbar.update(1)
                    continue
                
                # Get embedding
                embedding = get_document_embedding(text)
                
                if embedding is not None:
                    embeddings_data.append({
                        'document_name': doc_path.name,
                        'document_path': str(doc_path),
                        'text_length': len(text),
                        'embedding': embedding.tolist()
                    })
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\n⚠️  Error with {doc_path.name}: {e}")
                pbar.update(1)
                continue
    
    elapsed = time.time() - start_time
    
    # Save to CSV
    print(f"\n💾 Saving embeddings to CSV...")
    
    with open(DOCUMENT_EMBEDDINGS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['document_name', 'document_path', 'text_length'] + \
                 [f'dim_{i}' for i in range(len(embeddings_data[0]['embedding']))]
        writer.writerow(header)
        
        # Write data
        for item in embeddings_data:
            row = [item['document_name'], item['document_path'], item['text_length']] + \
                  item['embedding']
            writer.writerow(row)
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"✅ SUCCESS!")
    print(f"=" * 70)
    print(f"\n📊 Summary:")
    print(f"  • Documents embedded: {len(embeddings_data)}")
    print(f"  • Embedding dimension: 384 ({EMBEDDING_MODEL})")
    print(f"  • Total time: {elapsed/60:.1f} minutes")
    print(f"  • Avg time per doc: {elapsed/len(embeddings_data):.2f}s")
    print(f"\n📁 Output: {DOCUMENT_EMBEDDINGS_CSV}")
    print("=" * 70 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
