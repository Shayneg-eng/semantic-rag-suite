import os
"""
ShiftDim: Generate Content-Based Document Shifts for Long Documents
Handles chunking of documents over 2K token limit with intelligent boundary detection.
"""

import openai
import requests
import json
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# Configuration
# ============================================================================
POE_API_KEY = os.getenv("POE_API_KEY", "")
LLM_MODEL = "llama-3.1-8b-cs"

SOURCE_DOCS_PATH = Path(r"D:\LegalBench-RAG\corpus_flat")
OUTPUT_BASE_PATH = Path(r"c:\Coding\Code\RAG\ShiftDim\legal_shifts")

# Embedding configuration
EMBEDDING_BASE_URL = "http://localhost:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"

# Create output directory
OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Initialize LLM client
llm_client = openai.OpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

# ============================================================================
# Configuration
# ============================================================================

TOKEN_LIMIT = 2000
CHUNK_TARGET = 1500
SEARCH_WINDOW_INITIAL = 50


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English."""
    return len(text) // 4


def call_llm(prompt: str, temperature=0.0, max_tokens=500) -> str:
    """Call LLM via Poe API with deterministic settings."""
    try:
        chat = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=60.0
        )
        result = chat.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"    LLM call error: {type(e).__name__}: {e}")
        return "[Error generating shift]"


def embed_text(text: str) -> List[float]:
    """Generate embedding for text using nomic-embed-text via Ollama."""
    try:
        response = requests.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30.0
        )
        result = response.json()
        if "embedding" in result:
            return result["embedding"]
        else:
            print(f"    Embedding error: No embedding in response")
            return []
    except Exception as e:
        print(f"    Embedding error: {type(e).__name__}: {e}")
        return []


def find_chunk_boundary(text: str, target_pos: int, search_window: int = SEARCH_WINDOW_INITIAL) -> int:
    """
    Find intelligent chunk boundary starting from target_pos and searching backwards.
    
    Strategy:
    1. Search backwards from target_pos within search_window for line break + period
    2. If not found, search for just period
    3. If still not found, expand search_window by 10 tokens (40 chars) and retry
    4. Max iterations to prevent infinite loops
    """
    search_range = search_window * 4  # Convert tokens to chars
    start_pos = max(0, target_pos - search_range)
    search_text = text[start_pos:target_pos]
    
    # Strategy 1: Look for line break + period (highest priority)
    for i in range(len(search_text) - 1, -1, -1):
        if search_text[i] == '.' and (i + 1 < len(search_text) and search_text[i + 1] in '\n\r'):
            return start_pos + i + 1
    
    # Strategy 2: Look for just period
    for i in range(len(search_text) - 1, -1, -1):
        if search_text[i] == '.':
            return start_pos + i + 1
    
    # Strategy 3: Expand search window incrementally
    expanded_window = search_window + 10
    if expanded_window <= search_window + 100:  # Max 100 token expansion
        return find_chunk_boundary(text, target_pos, expanded_window)
    
    # Fallback: just split at target
    return target_pos


def chunk_document(text: str) -> List[str]:
    """
    Chunk document intelligently, keeping chunks under TOKEN_LIMIT.
    Uses find_chunk_boundary to respect sentence boundaries.
    """
    tokens = estimate_tokens(text)
    
    if tokens <= TOKEN_LIMIT:
        return [text]
    
    chunks = []
    pos = 0
    
    while pos < len(text):
        # Calculate target end position (target ~1K tokens, which is ~4K chars)
        target_end = pos + (CHUNK_TARGET * 4)
        
        if target_end >= len(text):
            # Last chunk
            chunks.append(text[pos:])
            break
        
        # Find intelligent boundary
        boundary = find_chunk_boundary(text, target_end)
        chunks.append(text[pos:boundary])
        pos = boundary
    
    return chunks


# ============================================================================
# LLM Transformation Functions (with output size constraints)
# ============================================================================

def entity_relationship_extraction(text: str) -> str:
    """Extract sentences as [Subject] [Verb] [Object] triplets."""
    prompt = f"""Extract the core facts from the following text as complete subject-verb-object triplets.

For EVERY main clause, extract: [Subject] [Verb] [Object/Result]

Rules:
- ALWAYS include complete triplets with all three parts filled in
- Use the specific nouns/entities from the text, not placeholders
- One triplet per line
- Format: subject verb object
- KEEP OUTPUT UNDER 2000 TOKENS
- Example triplet formats:
  * "Party A grants Party B rights"
  * "Contract term lasts 5 years"
  * "Termination occurs upon breach"

Do NOT extract incomplete fragments. Every line must be a complete triplet.

Text to process:
{text}

Output ONLY the complete triplets, one per line. No numbered lists, no headers, no explanations."""

    return call_llm(prompt, max_tokens=800)


def abstraction_level_normalization(text: str) -> str:
    """Replace specific instances with general category labels."""
    prompt = f"""Create a version of the following text where you replace all specific examples, proper nouns, and concrete instances with their general category labels.

For example:
- Replace "John Smith and Mary Jones" with "individuals"
- Replace "New York" with "jurisdiction"
- Replace "2024" with "temporal reference"

Replace ANY specific names, dates, companies, places, and concrete examples with their category. Be aggressive in generalizing.

KEEP OUTPUT UNDER 2000 TOKENS. Be concise.

Text to process:
{text}

Output ONLY the generalized version. No explanation or commentary."""

    return call_llm(prompt, max_tokens=800)


def modifier_qualifier_stripping(text: str) -> str:
    """Remove adjectives, adverbs, intensifiers, and hedging language."""
    prompt = f"""Remove all adjectives, adverbs, intensifiers, and hedging language from the following text.

Remove words like: substantial, material, reasonable, arguably, apparently, significant, etc.

Keep only:
- Adjectives that are definitional (like "material" in "material breach")
- Adjectives that are part of proper nouns
- Verbs and their core arguments (subject, object, etc.)
- Core nouns and essential prepositions

KEEP OUTPUT UNDER 2000 TOKENS. Be concise.

Text to process:
{text}

Output ONLY the stripped version. No explanation."""

    return call_llm(prompt, max_tokens=800)


def temporal_causal_structure_removal(text: str) -> str:
    """Strip temporal markers, causal connectors, and conditional statements."""
    prompt = f"""Remove all temporal markers and causal connectors from the following text.

Remove:
- Temporal markers: before, after, within, prior to, subsequently, etc.
- Causal connectors: because, therefore, as a result, caused, led to, etc.
- Conditional statements: if, unless, in case, provided that, etc.

Keep only the core state assertions. If "Because X happened, Y occurred", output "X occurred; Y occurred"

KEEP OUTPUT UNDER 2000 TOKENS. Be concise.

Text to process:
{text}

Output ONLY the modified version with pure state assertions. No explanation."""

    return call_llm(prompt, max_tokens=800)


def perspective_voice_normalization(text: str) -> str:
    """Standardize how claims are presented - remove perspective and convert to declarative."""
    prompt = f"""Normalize all claims to objective declarative form. Remove all perspective and voice markers:

- Convert reported speech: "It is stated that X" → "X"
- Convert rhetorical questions: "Isn't it clear that Y?" → "Y is true"
- Remove hedging: "The party may need to" → "The party needs to"
- Remove subjective assessment: "The agreement seems to require" → "The agreement requires"

Present everything as objective fact.

KEEP OUTPUT UNDER 2000 TOKENS. Be concise.

Text to process:
{text}

Output ONLY the normalized version. No explanation."""

    return call_llm(prompt, max_tokens=800)


def negation_isolation(text: str) -> str:
    """Convert negations to positive form."""
    prompt = f"""Rewrite the text so that every negated claim is converted to its positive form.

Rules:
- "X is NOT allowed" → "X is prohibited"
- "X does NOT include Y" → "X excludes Y"
- "X is NOT required" → "X is optional"
- Remove all "not", "no", "neither", "nor" constructions
- Express everything as positive assertions

KEEP OUTPUT UNDER 2000 TOKENS. Be concise.

Text to process:
{text}

Output ONLY the rewritten text with all negations converted to positive assertions. No explanations."""

    return call_llm(prompt, max_tokens=800)


def redundancy_collapse(text: str) -> str:
    """Identify and remove semantically equivalent statements."""
    prompt = f"""Identify all semantically equivalent statements in the text—claims that say essentially the same thing with different wording.

Keep ONLY the first occurrence of each unique claim. Remove all redundant restatements.

KEEP OUTPUT UNDER 2000 TOKENS. Be concise.

Text to process:
{text}

Output ONLY the deduplicated version with redundant claims removed. No explanation."""

    return call_llm(prompt, max_tokens=800)


def implicit_assumption_extraction(text: str) -> str:
    """Identify and make explicit only the NON-OBVIOUS implicit claims."""
    prompt = f"""Find NON-OBVIOUS implicit assumptions in the text. Do NOT list trivial or obvious assumptions.

Examples of NON-OBVIOUS assumptions worth extracting:
- Legal interpretations (why does this provision matter?)
- Intent implications (what must be true for this clause to work?)
- Risk assumptions (what happens if conditions change?)
- Precedent assumptions (what legal principles are invoked?)

Examples of TRIVIAL assumptions to IGNORE:
- "This document exists" (too obvious)
- "Parties want to execute agreements" (too obvious)

For each non-obvious assumption found, add one sentence explaining it.

KEEP OUTPUT UNDER 2000 TOKENS. Be concise.

Text to process:
{text}

Output ONLY the original text followed by a brief section listing 3-5 non-obvious implicit assumptions. Keep it concise."""

    return call_llm(prompt, max_tokens=800)


# ============================================================================
# Main Processing
# ============================================================================

def generate_shifts_for_chunk(chunk_num: int, chunk_text: str) -> dict:
    """Generate all 8 shifted versions of a document chunk in parallel."""
    
    shifts = {}
    
    print(f"  Generating shifts for chunk {chunk_num}...", end=" ", flush=True)
    
    # Define all shift functions
    shift_tasks = [
        ("1_entity_relationship_extraction", "Entity Relationship Extraction", entity_relationship_extraction),
        ("2_abstraction_normalization", "Abstraction Level Normalization", abstraction_level_normalization),
        ("3_modifier_qualifier_stripping", "Modifier Qualifier Stripping", modifier_qualifier_stripping),
        ("4_temporal_causal_removal", "Temporal Causal Removal", temporal_causal_structure_removal),
        ("5_perspective_voice_normalization", "Perspective Voice Normalization", perspective_voice_normalization),
        ("6_negation_isolation", "Negation Isolation", negation_isolation),
        ("7_redundancy_collapse", "Redundancy Collapse", redundancy_collapse),
        ("8_implicit_assumptions", "Implicit Assumptions", implicit_assumption_extraction),
    ]
    
    # Run all shifts in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(func, chunk_text): (key, name)
            for key, name, func in shift_tasks
        }
        
        for future in as_completed(futures):
            key, name = futures[future]
            try:
                shifts[key] = future.result()
            except Exception as e:
                print(f"\n    Error in {name}: {e}")
                shifts[key] = f"[Error generating shift: {e}]"
    
    print("✓")
    return shifts


def calculate_shift_metrics(original_text: str, shift_text: str) -> dict:
    """Calculate metrics comparing original to shifted text."""
    original_tokens = estimate_tokens(original_text)
    shift_tokens = estimate_tokens(shift_text)
    token_reduction = max(0, original_tokens - shift_tokens)
    reduction_pct = (token_reduction / original_tokens * 100) if original_tokens > 0 else 0
    
    return {
        "original_tokens": original_tokens,
        "shift_tokens": shift_tokens,
        "token_reduction": token_reduction,
        "reduction_percentage": round(reduction_pct, 1),
        "original_length": len(original_text),
        "shift_length": len(shift_text),
        "sentences_original": original_text.count('.') + original_text.count('!') + original_text.count('?'),
        "sentences_shift": shift_text.count('.') + shift_text.count('!') + shift_text.count('?'),
    }


def save_shifts_file(doc_name: str, chunk_num: int, chunks_total: int, chunk_text: str, shifts: dict) -> None:
    """Save all shifts with embeddings to file, computing embeddings in parallel."""
    output_file = OUTPUT_BASE_PATH / f"{doc_name}_chunk{chunk_num}_of_{chunks_total}_shifts.txt"
    
    shift_names = [
        "Entity and Relationship Extraction",
        "Abstraction Level Normalization",
        "Modifier and Qualifier Stripping",
        "Temporal and Causal Structure Removal",
        "Perspective and Voice Normalization",
        "Negation Isolation",
        "Redundancy and Repetition Collapse",
        "Implicit Assumption Extraction"
    ]
    
    # Generate embeddings in parallel
    shift_embeddings = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(embed_text, shifts[key]): key
            for key in sorted(shifts.keys())
        }
        for future in as_completed(futures):
            key = futures[future]
            shift_embeddings[key] = future.result()
    
    # Write file with embeddings
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header with document info
        f.write(f"DOCUMENT SHIFTS FOR {doc_name} (CHUNK {chunk_num}/{chunks_total})\n")
        f.write("=" * 80 + "\n\n")
        
        # Original unshifted chunk with metadata
        f.write("ORIGINAL UNSHIFTED CHUNK\n")
        f.write("-" * 80 + "\n\n")
        
        # Calculate and display metadata
        chunk_tokens = estimate_tokens(chunk_text)
        f.write(f"Metadata:\n")
        f.write(f"  Token Count: {chunk_tokens}\n")
        f.write(f"  Character Count: {len(chunk_text)}\n")
        f.write(f"  Sentence Count: {chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?')}\n")
        f.write(f"  Line Count: {len(chunk_text.splitlines())}\n")
        f.write(f"\n")
        
        # Original text
        f.write(f"Text:\n\n{chunk_text}\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Shift comparisons with metrics
        for (key, content), name in zip(sorted(shifts.items()), shift_names):
            embedding = shift_embeddings.get(key, [])
            embedding_str = json.dumps(embedding) if embedding else "[]"
            metrics = calculate_shift_metrics(chunk_text, content)
            
            f.write(f"\n{'=' * 80}\n")
            f.write(f"SHIFT {key.split('_')[0]}: {name}\n")
            f.write(f"EMBEDDING: {embedding_str}\n")
            f.write(f"{'=' * 80}\n\n")
            
            # Metrics for this shift
            f.write(f"Shift Metrics:\n")
            f.write(f"  Original Tokens: {metrics['original_tokens']}\n")
            f.write(f"  Shifted Tokens: {metrics['shift_tokens']}\n")
            f.write(f"  Token Reduction: {metrics['token_reduction']} ({metrics['reduction_percentage']}%)\n")
            f.write(f"  Original Sentences: {metrics['sentences_original']}\n")
            f.write(f"  Shifted Sentences: {metrics['sentences_shift']}\n")
            f.write(f"\n")
            
            f.write(content)
            f.write("\n\n")
    
    print(f"    ✓ Saved to {output_file.name}")


def main():
    """Main processing pipeline."""
    
    print("\n" + "=" * 80)
    print("ShiftDim: Chunked Document Shift Generation for Legal Corpus")
    print("=" * 80)
    
    # Get all legal documents
    doc_files = sorted(SOURCE_DOCS_PATH.glob("*.txt"))
    print(f"\nFound {len(doc_files)} documents\n")
    
    docs_processed = 0
    chunks_processed = 0
    docs_failed = 0
    
    for doc_idx, doc_file in enumerate(doc_files, 1):
        doc_name = doc_file.stem
        
        try:
            # Read document
            with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Estimate tokens
            token_count = estimate_tokens(text)
            print(f"[{doc_idx}/{len(doc_files)}] {doc_name} ({token_count:,} tokens)")
            
            # Chunk if necessary
            chunks = chunk_document(text)
            print(f"  → {len(chunks)} chunk(s)")
            
            # Generate shifts for each chunk
            for chunk_num, chunk_text in enumerate(chunks, 1):
                shifts = generate_shifts_for_chunk(chunk_num, chunk_text)
                save_shifts_file(doc_name, chunk_num, len(chunks), chunk_text, shifts)
                chunks_processed += 1
            
            print(f"  ✓ COMPLETE\n")
            docs_processed += 1
            
        except Exception as e:
            print(f"  ✗ FAILED ({type(e).__name__}: {str(e)[:60]})\n")
            docs_failed += 1
    
    print("=" * 80)
    print(f"Complete: {docs_processed} docs, {chunks_processed} chunks, {docs_failed} failed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
