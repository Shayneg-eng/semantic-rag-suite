import os
import csv
import numpy as np
from ollama import embed
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

def get_embedding(text):
    """Get embedding for text using ollama"""
    try:
        response = embed(model="nomic-embed-text", input=text)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"Error embedding: {e}")
        return None

def chunk_text(text, chunk_size=100):
    """Split text into chunks of approximately chunk_size words"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def process_document(args):
    """Process a single document and return its chunks with embeddings"""
    doc_name, doc_path = args
    chunks_data = []
    
    try:
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Split into chunks
        chunks = chunk_text(text, chunk_size=100)
        
        # Embed each chunk
        for chunk_idx, chunk_text_content in enumerate(chunks):
            embedding = get_embedding(chunk_text_content)
            
            if embedding is not None:
                chunks_data.append({
                    'document_name': doc_name,
                    'chunk_index': chunk_idx,
                    'chunk_text': chunk_text_content[:200],
                    'text_length': len(chunk_text_content),
                    'embedding': embedding
                })
        
        return chunks_data
    
    except Exception as e:
        print(f"Error processing {doc_name}: {e}")
        return []

# Find all documents
corpus_dir = "LegalBench-RAG\\corpus_flat"
if not os.path.exists(corpus_dir):
    print(f"Corpus directory not found: {corpus_dir}")
    exit(1)

documents = []
for root, dirs, files in os.walk(corpus_dir):
    for file in files:
        if file.endswith('.txt'):
            full_path = os.path.join(root, file)
            documents.append((file, full_path))

print(f"Found {len(documents)} documents")
print(f"Creating chunks of 100 words with parallelization...\n")

# Create output CSV
output_file = "embeddings/chunk_embeddings.csv"
os.makedirs("embeddings", exist_ok=True)

# Use ThreadPoolExecutor for parallel processing
num_workers = 8
total_chunks = 0
all_chunks_data = []

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Submit all tasks
    futures = {executor.submit(process_document, doc): doc[0] for doc in documents}
    
    # Process completed tasks with progress bar
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
        doc_name = futures[future]
        try:
            chunks_data = future.result()
            all_chunks_data.extend(chunks_data)
            total_chunks += len(chunks_data)
        except Exception as e:
            print(f"Error with {doc_name}: {e}")

print(f"\nWriting {total_chunks} chunks to CSV...")

# Write all chunks to CSV in one go (more efficient)
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['document_name', 'chunk_index', 'chunk_text', 'text_length'] + [f'dim_{i}' for i in range(768)]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for chunk_data in tqdm(all_chunks_data, desc="Writing to CSV"):
        row = {
            'document_name': chunk_data['document_name'],
            'chunk_index': chunk_data['chunk_index'],
            'chunk_text': chunk_data['chunk_text'],
            'text_length': chunk_data['text_length']
        }
        
        # Add embedding dimensions
        for i, val in enumerate(chunk_data['embedding']):
            row[f'dim_{i}'] = val
        
        writer.writerow(row)

print(f"\n✅ Completed!")
print(f"Total chunks created: {total_chunks}")
print(f"Saved to: {output_file}")
