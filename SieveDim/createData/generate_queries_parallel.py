import os
"""
Generate 2000 query-chunk pairs in parallel (20 at a time)
No duplicate chunks, no adjacent chunks allowed
"""

import json
import csv
import random
from pathlib import Path
import openai
import ollama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


def load_all_chunk_files(corpus_folder):
    """
    Load all chunk filenames from corpus
    Returns list of chunk filenames
    """
    stats_path = Path(corpus_folder) / "chunking_stats.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Extract base filenames and chunk counts
    chunk_files = []
    for item in stats['files_processed']:
        base_file = item['file'].replace('.txt', '')
        chunks_created = item['chunks_created']
        
        # Create chunk filenames for each chunk
        for chunk_num in range(1, chunks_created + 1):
            chunk_filename = f"{base_file}___chunk_{chunk_num:04d}.txt"
            chunk_files.append(chunk_filename)
    
    return chunk_files


def is_adjacent(chunk_id, used_chunks, all_chunks):
    """
    Check if chunk_id is adjacent to any chunk in used_chunks
    Adjacent means the index differs by exactly 1
    """
    for used_id in used_chunks:
        if abs(chunk_id - used_id) == 1:
            return True
    return False


def select_random_non_adjacent_chunk(all_chunks, used_chunks):
    """
    Select a random chunk that:
    1. Hasn't been used yet
    2. Is not adjacent to any used chunk
    """
    available = [
        i for i in range(len(all_chunks))
        if i not in used_chunks and not is_adjacent(i, used_chunks, all_chunks)
    ]
    
    if not available:
        return None, None
    
    chunk_idx = random.choice(available)
    return chunk_idx, all_chunks[chunk_idx]


def generate_single_query(chunk_filename, corpus_folder, client):
    """
    Generate a single query for a given chunk
    """
    chunk_path = Path(corpus_folder) / chunk_filename
    
    try:
        if not chunk_path.exists():
            return None
        
        with open(chunk_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content.strip():
            return None
        
        content_preview = content[:1000]
        
        # Create prompt for LLM to generate RAG-style search queries
        prompt = f"""You are an expert at creating natural search queries for a Retrieval-Augmented Generation (RAG) system that indexes legal documents.

Given the following NDA/contract excerpt, generate ONE realistic search query that someone would use to find this information.

The query should:
- Be a natural, concise search query (like what a user would type into a search bar)
- Be 3-10 words long
- Focus on key concepts and entities from the text
- Avoid being too formal - sound like natural language
- Be specific enough to retrieve relevant documents
- NOT be a complete question - be more keyword-focused

Document excerpt:
---
{content_preview}
---

Generate ONLY the search query, no explanation. Output just the query."""
        
        # Call LLM
        chat = client.chat.completions.create(
            model="llama-3.1-8b-cs",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        query = chat.choices[0].message.content.strip()
        
        return {
            'query': query,
            'correct_document': chunk_filename
        }
        
    except Exception as e:
        print(f"Error processing {chunk_filename}: {e}")
        return None


def embed_query(query_text):
    """
    Embed a single query text using Ollama
    """
    try:
        response = ollama.embed(
            model='nomic-embed-text',
            input=query_text,
        )
        return response.embeddings[0] if response.embeddings else None
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None


def generate_queries_parallel(corpus_folder, num_samples=2000, batch_size=20, output_file="query_document_pairs.csv"):
    """
    Generate queries in parallel batches
    
    Args:
        corpus_folder: Path to corpus_chunks directory
        num_samples: Total number of query pairs to generate (2000)
        batch_size: Number of parallel queries (20)
        output_file: Output CSV file name
    """
    
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key=os.getenv("POE_API_KEY", ""),
        base_url="https://api.poe.com/v1",
    )
    
    # Load all chunks
    print("Loading all chunk files...")
    all_chunks = load_all_chunk_files(corpus_folder)
    print(f"Found {len(all_chunks)} total chunks")
    
    # Prepare CSV
    output_path = Path(corpus_folder).parent / output_file
    csv_file = open(output_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=['query', 'correct_document', 'query_embedding'])
    csv_writer.writeheader()
    
    used_chunks = set()
    generated_count = 0
    
    print(f"\nGenerating {num_samples} query-document pairs in batches of {batch_size}...\n")
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        while generated_count < num_samples:
            # Prepare batch of chunks
            batch_chunks = []
            batch_indices = []
            
            for _ in range(batch_size):
                if generated_count >= num_samples:
                    break
                
                chunk_idx, chunk_file = select_random_non_adjacent_chunk(all_chunks, used_chunks)
                
                if chunk_idx is None:
                    print(f"⚠️  No more non-adjacent chunks available. Generated {generated_count} pairs.")
                    break
                
                batch_chunks.append(chunk_file)
                batch_indices.append(chunk_idx)
                generated_count += 1
            
            if not batch_chunks:
                break
            
            # Submit batch tasks
            futures = [
                executor.submit(generate_single_query, chunk, corpus_folder, client)
                for chunk in batch_chunks
            ]
            
            # Collect generated queries
            batch_results = []
            batch_queries = []
            
            for future, chunk_idx in zip(as_completed(futures), batch_indices):
                try:
                    result = future.result()
                    if result:
                        batch_results.append((result, chunk_idx))
                        batch_queries.append(result['query'])
                    else:
                        print(f"✗ Failed to generate query for chunk {chunk_idx}")
                except Exception as e:
                    print(f"✗ Generation error: {e}")
            
            # Submit embedding tasks in parallel
            if batch_queries:
                print(f"\n📊 Embedding {len(batch_queries)} queries in parallel...")
                embedding_futures = [
                    executor.submit(embed_query, query)
                    for query in batch_queries
                ]
                
                # Collect embedding results
                embeddings = []
                for embedding_future in as_completed(embedding_futures):
                    try:
                        embedding = embedding_future.result()
                        embeddings.append(embedding)
                    except Exception as e:
                        print(f"✗ Embedding error: {e}")
                        embeddings.append(None)
                
                # Write results with embeddings
                for (result, chunk_idx), embedding in zip(batch_results, embeddings):
                    if embedding:
                        result['query_embedding'] = json.dumps(embedding)
                        csv_writer.writerow(result)
                        csv_file.flush()
                        used_chunks.add(chunk_idx)
                        print(f"✓ [{len(used_chunks)}/{num_samples}] {result['correct_document'][:50]}... - Query: {result['query'][:60]}...")
                    else:
                        print(f"✗ Skipped - embedding failed for: {result['query'][:60]}...")
            
            # Rate limiting between batches
            time.sleep(0.5)
    
    csv_file.close()
    print(f"\n✅ Generated {len(used_chunks)} query-document pairs")
    print(f"📁 Saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    corpus_path = "corpus_chunks"
    generate_queries_parallel(corpus_path, num_samples=2000, batch_size=20)
