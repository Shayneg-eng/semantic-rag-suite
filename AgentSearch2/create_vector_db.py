import os
import json
from pathlib import Path
import ollama
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_summary_text(summary_file_path):
    """Extract summary text, excluding the [TOKENS: X] header."""
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove [TOKENS: X] header if present
    lines = content.split('\n')
    if lines[0].startswith('[TOKENS:'):
        # Skip first line and empty line
        summary_text = '\n'.join(lines[2:]).strip()
    else:
        summary_text = content.strip()
    
    return summary_text

def embed_summary(summary_file_path, summary_name):
    """Create embedding for a single summary."""
    try:
        summary_text = extract_summary_text(summary_file_path)
        
        # Call ollama to generate embedding
        response = ollama.embed(
            model='nomic-embed-text',
            input=summary_text,
        )
        
        embedding = response['embeddings'][0]  # Get first (and only) embedding
        
        return True, summary_name, summary_text, embedding, None
    except Exception as e:
        return False, summary_name, None, None, str(e)

def main():
    summaries_folder = "summaries"
    db_output = "vector_db"
    
    # Create output folder
    Path(db_output).mkdir(exist_ok=True)
    
    # Get all summary files
    summaries_path = Path(summaries_folder)
    summary_files = list(summaries_path.glob("*_summary.txt"))
    
    print(f"Found {len(summary_files)} summaries to embed")
    print(f"Using model: nomic-embed-text")
    print(f"Saving to: {db_output}/\n")
    
    # Database structure
    vector_db = {
        "metadata": {
            "model": "nomic-embed-text",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_documents": len(summary_files)
        },
        "documents": {}
    }
    
    completed = 0
    failed = 0
    failed_docs = []
    start_time = time.time()
    
    # Process with parallel API calls
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(embed_summary, summary_file, summary_file.name): summary_file for summary_file in summary_files}
        
        for future in as_completed(futures):
            try:
                success, summary_name, summary_text, embedding, error = future.result()
                
                if success:
                    # Extract original document name
                    doc_name = summary_name.replace('_summary.txt', '.txt')
                    
                    # Store in database
                    vector_db["documents"][doc_name] = {
                        "summary_file": summary_name,
                        "summary_text": summary_text,
                        "embedding": embedding,  # Will be converted to list for JSON
                        "embedding_dim": len(embedding)
                    }
                    
                    completed += 1
                    elapsed = time.time() - start_time
                    print(f"[{completed}/{len(summary_files)}] ✓ {doc_name}")
                    print(f"   Embedding dim: {len(embedding)}")
                    print(f"   Time: {elapsed:.1f}s\n")
                else:
                    failed += 1
                    failed_docs.append((summary_name, error))
                    print(f"[{completed + failed}/{len(summary_files)}] ✗ {summary_name}")
                    print(f"   Error: {error}\n")
                
            except Exception as e:
                failed += 1
                print(f"[{completed + failed}/{len(summary_files)}] ✗ Error: {e}\n")
    
    # Save database to JSON (embeddings as lists)
    print("\nSaving vector database...")
    db_json_path = Path(db_output) / "vector_db.json"
    
    # Convert numpy arrays to lists for JSON serialization
    db_for_json = {
        "metadata": vector_db["metadata"],
        "documents": {}
    }
    
    for doc_name, doc_data in vector_db["documents"].items():
        db_for_json["documents"][doc_name] = {
            "summary_file": doc_data["summary_file"],
            "summary_text": doc_data["summary_text"],
            "embedding": doc_data["embedding"] if isinstance(doc_data["embedding"], list) else doc_data["embedding"].tolist(),
            "embedding_dim": doc_data["embedding_dim"]
        }
    
    with open(db_json_path, 'w', encoding='utf-8') as f:
        json.dump(db_for_json, f)
    
    # Also save embeddings as numpy file for efficient search
    print("Creating FAISS-compatible index...")
    embeddings_array = np.array([
        vector_db["documents"][doc_name]["embedding"] 
        for doc_name in sorted(vector_db["documents"].keys())
    ])
    
    doc_names = sorted(vector_db["documents"].keys())
    
    np.save(Path(db_output) / "embeddings.npy", embeddings_array)
    
    # Save document order mapping
    with open(Path(db_output) / "doc_names.json", 'w') as f:
        json.dump(doc_names, f)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"COMPLETE: {completed}/{len(summary_files)} documents embedded")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(summary_files):.1f}s per document)")
    print(f"Success: {completed}, Failed: {failed}")
    print(f"\nDatabase saved to:")
    print(f"  - {db_json_path}")
    print(f"  - {Path(db_output) / 'embeddings.npy'}")
    print(f"  - {Path(db_output) / 'doc_names.json'}")
    print("="*60)
    
    if failed_docs:
        print("\nFailed documents:")
        for doc_name, error in failed_docs:
            print(f"  - {doc_name}: {error}")

if __name__ == "__main__":
    main()
