import os
"""
RAG Retrieval Test using Directional Shift Algorithm
Adapted from shift_test.py for legal corpus chunks
"""

import requests
import openai
import numpy as np
from pathlib import Path
from typing import List, Dict
import re
import hashlib
import json
from tqdm import tqdm

# Configuration
POE_API_KEY = os.getenv("POE_API_KEY", "")
LLM_MODEL = "llama-3.1-8b-cs"
EMBEDDING_BASE_URL = "http://localhost:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"
SHIFTS_DIR = Path(r"c:\Coding\Code\RAG\ShiftDim\legal_shifts")
CACHE_PATH = Path(r"c:\Coding\Code\RAG\ShiftDim\.cache_legal")
CACHE_PATH.mkdir(exist_ok=True)

llm_client = openai.OpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

SHIFT_NAMES = [
    "Entity and Relationship Extraction",
    "Abstraction Level Normalization",
    "Modifier and Qualifier Stripping",
    "Temporal and Causal Structure Removal",
    "Perspective and Voice Normalization",
    "Negation Isolation",
    "Redundancy and Repetition Collapse",
    "Implicit Assumption Extraction",
]

# ============================================================================
# Caching Layer (from shift_test.py)
# ============================================================================

def _get_text_hash(text: str) -> str:
    """Get MD5 hash of text for cache key."""
    return hashlib.md5(text.encode()).hexdigest()

def _get_embedding_cache_path(text_hash: str) -> Path:
    """Get cache file path for embedding."""
    return CACHE_PATH / f"embedding_{text_hash}.json"

def _get_shift_direction_cache_path(text_hash: str, shift_hash: str) -> Path:
    """Get cache file path for shift direction."""
    return CACHE_PATH / f"shift_dir_{text_hash}_{shift_hash}.json"

def _get_alignment_cache_path(query_hash: str, doc_hash: str, query_shifts_hash: str, doc_shifts_hash: str) -> Path:
    """Get cache file path for alignment score."""
    combined = f"{query_hash}_{doc_hash}_{query_shifts_hash}_{doc_shifts_hash}"
    combined_hash = hashlib.md5(combined.encode()).hexdigest()
    return CACHE_PATH / f"alignment_{combined_hash}.json"

def _save_cache(path: Path, data: any) -> None:
    """Save data to cache file."""
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Cache save error: {e}")

def _load_cache(path: Path) -> any:
    """Load data from cache file."""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")
        return None

# ============================================================================
# Embedding Functions
# ============================================================================

def embed_text(text: str) -> List[float]:
    """Embed text using nomic-embed-text with caching."""
    text_hash = _get_text_hash(text)
    cache_path = _get_embedding_cache_path(text_hash)
    
    # Try to load from cache
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached
    
    try:
        response = requests.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30.0
        )
        result = response.json()
        if "embedding" in result:
            embedding = result["embedding"]
            _save_cache(cache_path, embedding)
            return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
    return None

# ============================================================================
# File Parsing
# ============================================================================

def get_all_shift_files() -> List[Path]:
    """Get all shift files."""
    return sorted(list(SHIFTS_DIR.glob("*_shifts.txt")))

def parse_shift_file(file_path: Path) -> dict:
    """Parse shift file and extract original chunk, shift texts, AND pre-computed embeddings."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract original chunk
        original_start = content.find("ORIGINAL UNSHIFTED CHUNK")
        if original_start == -1:
            return None
        
        text_marker = content.find("Text:", original_start)
        if text_marker == -1:
            return None
        
        text_start = text_marker + 6
        text_end = content.find("=" * 80, text_start)
        if text_end == -1:
            return None
        
        original_text = content[text_start:text_end].strip()
        
        # Extract shift texts AND embeddings
        shift_texts = {}
        shift_embeddings = {}
        
        for shift_num in range(1, 9):
            shift_header = f"SHIFT {shift_num}:"
            shift_start = content.find(shift_header)
            if shift_start == -1:
                continue
            
            # Extract embedding vector
            emb_start = content.find("EMBEDDING: [", shift_start)
            if emb_start == -1:
                continue
            
            emb_start += 12
            emb_end = content.find("]", emb_start)
            emb_str = content[emb_start:emb_end]
            
            try:
                embedding = [float(x.strip()) for x in emb_str.split(',')]
                shift_embeddings[shift_num] = np.array(embedding, dtype=np.float32)
            except:
                continue
            
            # Extract shift text
            metrics_end = content.find("Shifted Sentences:", shift_start)
            if metrics_end == -1:
                continue
            
            text_start = content.find("\n", metrics_end) + 1
            next_marker = content.find("=" * 80, text_start)
            if next_marker == -1:
                next_marker = len(content)
            
            shift_texts[shift_num] = content[text_start:next_marker].strip()
        
        if not shift_embeddings:
            return None
        
        return {
            "file": file_path.name,
            "original": original_text,
            "shift_texts": shift_texts,
            "shift_embeddings": shift_embeddings  # Pre-computed embeddings!
        }
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None

# ============================================================================
# Directional Shift Algorithm (from shift_test.py)
# ============================================================================

def get_shift_direction(text: str, shift_text: str) -> np.ndarray:
    """
    Get the direction that embedding shifts when transforming.
    Returns normalized delta vector (unit length).
    With caching.
    """
    text_hash = _get_text_hash(text)
    shift_hash = _get_text_hash(shift_text)
    cache_path = _get_shift_direction_cache_path(text_hash, shift_hash)
    
    # Try to load from cache
    cached = _load_cache(cache_path)
    if cached is not None:
        return np.array(cached) if cached != "None" else None
    
    # Compute shift direction
    original = embed_text(text)
    if original is None:
        _save_cache(cache_path, "None")
        return None
    
    transformed_emb = embed_text(shift_text)
    if transformed_emb is None:
        _save_cache(cache_path, "None")
        return None
    
    # Delta: how embedding changed
    delta = np.array(transformed_emb) - np.array(original)
    
    # Normalize to unit vector (direction only)
    norm = np.linalg.norm(delta)
    if norm == 0:
        _save_cache(cache_path, "None")
        return None
    
    result = (delta / norm).tolist()
    _save_cache(cache_path, result)
    return np.array(result)

def fast_directional_alignment_with_embeddings(query_embeddings: Dict[int, np.ndarray], doc_embeddings: Dict[int, np.ndarray]) -> float:
    """
    Compute alignment score using pre-computed shift embeddings.
    Much faster than computing embeddings on the fly!
    
    query_embeddings: {shift_num -> embedding vector}
    doc_embeddings: {shift_num -> embedding vector}
    """
    scores = []
    
    for shift_num in range(1, 9):
        if shift_num not in query_embeddings or shift_num not in doc_embeddings:
            continue
        
        q_emb = query_embeddings[shift_num]
        d_emb = doc_embeddings[shift_num]
        
        # Normalize vectors to unit length
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        d_norm = d_emb / (np.linalg.norm(d_emb) + 1e-8)
        
        # Cosine similarity = dot product of normalized vectors
        similarity = float(np.dot(q_norm, d_norm))
        scores.append(similarity)
    
    if not scores:
        return None
    
    return np.mean(scores)

# ============================================================================
# Query Transformation
# ============================================================================

def generate_question(text: str) -> str:
    """Generate a question about the text using Llama."""
    text_preview = text[:2000] if len(text) > 2000 else text
    
    prompt = f"""Based on the following legal text, generate ONE specific, detailed question that could only be answered by reading this exact chunk. The question should be answerable from the content provided.

Text:
{text_preview}

Generate only the question, nothing else."""
    
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
            timeout=30.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return None

def apply_shifts_to_query(query: str) -> Dict[int, str]:
    """Apply all 8 shifts to query using LLM."""
    shifts = {}
    
    prompts = [
        (1, f"Extract entity-relationship triplets from this legal question: {query}"),
        (2, f"Generalize and abstract this legal question: {query}"),
        (3, f"Strip modifiers from this legal question: {query}"),
        (4, f"Remove temporal markers from this legal question: {query}"),
        (5, f"Normalize to declarative form: {query}"),
        (6, f"Convert to positive only: {query}"),
        (7, f"Collapse redundancy in: {query}"),
        (8, f"Extract legal assumptions from: {query}"),
    ]
    
    print("  Applying shifts to query:")
    for shift_num, prompt in prompts:
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                timeout=30.0
            )
            shifts[shift_num] = response.choices[0].message.content.strip()
            print(f"    Shift {shift_num}/8 ✓")
        except Exception as e:
            print(f"    Error generating shift {shift_num}: {e}")
            shifts[shift_num] = query  # Fallback to original query
    
    return shifts

# ============================================================================
# Main Test
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("RAG RETRIEVAL TEST: Directional Shift Algorithm (Legal Corpus)")
    print("=" * 80)
    
    # Load all documents
    print("\nLoading all shift files...")
    files = get_all_shift_files()
    print(f"Found {len(files)} shift files")
    
    all_docs = {}
    for file_path in files:
        doc_data = parse_shift_file(file_path)
        if doc_data:
            all_docs[file_path.name] = doc_data
    
    print(f"Successfully parsed {len(all_docs)} documents")
    
    if not all_docs:
        print("No documents found!")
        return
    
    # Pick random source chunk
    import random
    source_file = random.choice(list(all_docs.keys()))
    source_data = all_docs[source_file]
    
    print(f"\n{'='*80}")
    print(f"SOURCE: {source_file}")
    print(f"{'='*80}")
    
    original_text = source_data["original"]
    source_shifts = source_data["shift_texts"]
    
    print(f"\nOriginal chunk ({len(original_text)} chars):")
    print("-" * 80)
    preview_len = min(400, len(original_text))
    print(original_text[:preview_len] + ("..." if len(original_text) > preview_len else ""))
    print("-" * 80)
    
    # Generate question
    print("\nGenerating question from chunk...")
    question = generate_question(original_text)
    if not question:
        print("Failed to generate question!")
        return
    
    print(f"\nGenerated Question:\n{question}")
    
    # Apply shifts to question
    print("\nApplying 8 shifts to question...")
    question_shifts = apply_shifts_to_query(question)
    print("* Question shifted in all 8 ways\n")
    
    # Pre-compute question embeddings (fast - embeddings already cached)
    print("Computing question embeddings...")
    question_embeddings = {}
    text_hash = _get_text_hash(question)
    
    for shift_num in range(1, 9):
        if shift_num not in question_shifts:
            continue
        shifted_q = question_shifts[shift_num]
        if not shifted_q or shifted_q.startswith("[Error"):
            continue
        
        # Try to load from cache first
        shift_hash = _get_text_hash(shifted_q)
        cache_path = _get_embedding_cache_path(shift_hash)
        cached = _load_cache(cache_path)
        
        if cached is not None and cached != "None":
            question_embeddings[shift_num] = np.array(cached, dtype=np.float32)
        else:
            # Embed if not cached
            emb = embed_text(shifted_q)
            if emb is not None:
                question_embeddings[shift_num] = np.array(emb, dtype=np.float32)
    
    print(f"Question has {len(question_embeddings)}/8 embeddings")
    
    print("Computing directional alignment with all documents...")
    alignments = {}
    
    for doc_file, doc_data in tqdm(all_docs.items(), total=len(all_docs), desc="Alignment", unit="doc"):
        # Use pre-computed embeddings from shift file!
        if "shift_embeddings" not in doc_data or not doc_data["shift_embeddings"]:
            continue
        
        doc_embeddings = doc_data["shift_embeddings"]
        
        # Fast computation with pre-computed embeddings
        alignment = fast_directional_alignment_with_embeddings(question_embeddings, doc_embeddings)
        if alignment is not None:
            alignments[doc_file] = alignment
    
    print(f"✓ Computed alignments for {len(alignments)} documents\n")
    
    # Rank results
    ranked_results = sorted(alignments.items(), key=lambda x: x[1], reverse=True)
    
    # Display top 10
    print("=" * 80)
    print("TOP 10 MATCHES (Directional Shift Algorithm):")
    print("=" * 80)
    
    for i, (doc_file, score) in enumerate(ranked_results[:10], 1):
        is_source = " ← SOURCE" if doc_file == source_file else ""
        print(f"{i}. {doc_file}")
        print(f"   Alignment Score: {score:+.4f}{is_source}")
    
    # Check source ranking
    source_rank = next((i for i, (doc, _) in enumerate(ranked_results, 1) if doc == source_file), None)
    
    print("\n" + "=" * 80)
    if source_rank:
        print(f"* Source chunk found at rank #{source_rank}/{len(alignments)}")
        if source_rank == 1:
            print("  Perfect! Source is top match.")
        elif source_rank <= 3:
            print("  Excellent! Source in top 3.")
        elif source_rank <= 10:
            print("  Good! Source in top 10.")
        else:
            print(f"  Source ranked lower than expected.")
    else:
        print("x Source chunk not in results")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()