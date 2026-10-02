import os
import json
import ollama
from pathlib import Path
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configuration
CORPUS_DIR = os.path.join(os.path.dirname(__file__), '..', 'corpus_chunks')
OUTPUT_DB = os.path.join(os.path.dirname(__file__), '..', 'vector_db.pkl')
EMBEDDING_MODEL = 'nomic-embed-text'
NUM_WORKERS = 20

def split_at_sentence(content, target_pos):
    """
    Find a sentence break near target_pos and split there.
    Returns tuple: (first_part, second_part)
    Tries to find a sentence boundary (period, question mark, exclamation) within ±20% of target_pos.
    """
    search_start = max(0, int(target_pos * 0.8))
    search_end = min(len(content), int(target_pos * 1.2))
    
    # Look for sentence endings in the search range
    sentence_endings = ['.', '!', '?', ':\n', ';\n']
    best_pos = target_pos
    
    for i in range(search_end, search_start, -1):
        if i < len(content):
            # Check for sentence ending patterns
            for ending in sentence_endings:
                if content[i:i+len(ending)].startswith(ending):
                    # Found a sentence ending, use it
                    best_pos = i + len(ending)
                    # Skip trailing whitespace
                    while best_pos < len(content) and content[best_pos] in ' \n\t':
                        best_pos += 1
                    return content[:best_pos], content[best_pos:]
    
    # If no sentence boundary found, just split at target
    return content[:target_pos], content[target_pos:]

def embed_with_fallback(content, doc_filename):
    """
    Try to embed content. If it fails due to length, recursively split and embed chunks.
    Returns list of (content_chunk, success_flag) tuples.
    """
    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=content,
        )
        return [(content, True)]
    except Exception as e:
        error_msg = str(e).lower()
        if 'too long' in error_msg or 'context' in error_msg or 'length' in error_msg:
            # Document is too long, split it
            tqdm.write(f"  {doc_filename}: Too long ({len(content):,} chars), splitting...")
            
            # Split roughly in half with 20% overlap
            midpoint = len(content) // 2
            overlap_size = len(content) // 5
            
            first_part, second_part = split_at_sentence(content, midpoint - overlap_size)
            
            # Add overlap to second part
            if len(first_part) > overlap_size:
                second_part = first_part[-overlap_size:] + second_part
            
            tqdm.write(f"    Part 1: {len(first_part):,} chars, Part 2: {len(second_part):,} chars")
            
            # Recursively process each part
            results = []
            results.extend(embed_with_fallback(first_part, f"{doc_filename}_part1"))
            results.extend(embed_with_fallback(second_part, f"{doc_filename}_part2"))
            return results
        else:
            # Different error, re-raise
            raise

def embed_document(filepath):
    """Embed a single document and return filename with embeddings."""
    filename = filepath.name
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()
        
        # Try to embed with fallback splitting
        chunks = embed_with_fallback(original_content, filename)
        
        results = []
        for chunk_content, success in chunks:
            if success:
                # Get embedding for this chunk
                response = ollama.embed(
                    model=EMBEDDING_MODEL,
                    input=chunk_content,
                )
                
                results.append((filename, {
                    'content': chunk_content,
                    'embedding': response['embeddings'][0],
                    'model': EMBEDDING_MODEL,
                    'is_chunk': len(chunks) > 1,
                }))
        
        return filename, results, None
    except Exception as e:
        return filename, [], str(e)

def main():
    """Create vector database with parallel embedding and intelligent document splitting."""
    print("Creating Vector Database from Corpus Chunks")
    print("=" * 50)
    
    corpus_path = Path(CORPUS_DIR)
    txt_files = list(corpus_path.glob('*.txt'))
    
    print(f"Found {len(txt_files)} text files in corpus_chunks")
    
    if not txt_files:
        print("No documents found in corpus_chunks directory!")
        return
    
    vector_db = {}
    failed_docs = []
    total_chunks = 0
    
    print(f"\nEmbedding with {NUM_WORKERS} parallel workers...")
    print("Long documents will be intelligently split at sentence boundaries\n")
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(embed_document, file_path): file_path.name 
            for file_path in txt_files
        }
        
        # Process results as they complete
        with tqdm(total=len(txt_files), desc="Embedding") as pbar:
            for future in as_completed(futures):
                filename, results, error = future.result()
                
                if error:
                    failed_docs.append((filename, error))
                    tqdm.write(f"❌ Error embedding {filename}: {error}")
                elif results:
                    for orig_filename, embedding_data in results:
                        vector_db[orig_filename] = embedding_data
                        total_chunks += 1
                
                pbar.update(1)
                
                # Save after each document completes
                with open(OUTPUT_DB, 'wb') as f:
                    pickle.dump(vector_db, f)
    
    print(f"\n✅ Vector database saved to: {OUTPUT_DB}")
    print(f"   Total entries: {len(vector_db)}")
    print(f"   Total chunks created: {total_chunks}")
    
    # Count documents with chunks
    chunked_docs = sum(1 for doc in vector_db.values() if doc.get('is_chunk', False))
    if chunked_docs > 0:
        print(f"   Documents split into chunks: {chunked_docs}")
    
    if failed_docs:
        print(f"\n⚠️  Failed to embed {len(failed_docs)} documents:")
        for filename, error in failed_docs:
            print(f"   - {filename}: {error}")

if __name__ == '__main__':
    main()
