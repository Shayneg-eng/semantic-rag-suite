import ollama
import openai
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import *

# Initialize OpenAI client for Poe
client = openai.OpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

def load_documents(directory: str) -> Dict[str, str]:
    """Load all .txt documents from the source directory"""
    docs = {}
    doc_dir = Path(directory)
    
    if not doc_dir.exists():
        print(f"Error: Directory '{directory}' not found!")
        return docs
    
    file_paths = sorted(doc_dir.glob("*.txt"))
    
    # Limit documents if MAX_DOCUMENTS is set
    if MAX_DOCUMENTS is not None:
        file_paths = file_paths[:MAX_DOCUMENTS]
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    docs[file_path.name] = content
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    return docs

def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for text using Ollama"""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def apply_shift(text: str, shift_name: str) -> str:
    """Apply a semantic shift to text using deterministic word replacement"""
    try:
        # Map shift names to prompts and replacement words
        shift_config = {
            "remove_adjectives": (ADJECTIVE_EXTRACTION_PROMPT, ADJECTIVE_REPLACEMENT_WORD),
        }
        
        if shift_name not in shift_config:
            return text
        
        prompt_template, replacement_word = shift_config[shift_name]
        prompt = prompt_template.format(text=text)
        
        # Get list of words to replace
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=0.0000000001,
            seed=42
        )
        
        words_to_replace = response.choices[0].message.content.strip()
        
        # Parse the comma-separated list
        words = [w.strip().lower() for w in words_to_replace.split(',') if w.strip()]
        
        # Replace each word in the text (case-insensitive)
        result = text
        for word in words:
            # Use word boundaries to avoid partial replacements
            import re
            # Handle empty replacement string (for articles)
            result = re.sub(r'\b' + re.escape(word) + r'\b', replacement_word, result, flags=re.IGNORECASE)
        
        # Clean up extra whitespace from removals
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    except Exception as e:
        print(f"Shift error: {e}")
        return text

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if v1 is None or v2 is None:
        return 0.0
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    return dot_product / norm_product

def process_document(doc_id: str, doc_text: str) -> Dict:
    """Process document: generate embeddings for original and all shifts"""
    print(f"  Processing {doc_id}...")
    
    result = {
        "doc_id": doc_id,
        "original_embedding": None,
        "shifts": {}
    }
    
    # Get original embedding
    original_emb = get_embedding(doc_text)
    result["original_embedding"] = original_emb
    
    # Apply each shift and store shift vector
    for shift_name, shift_prompt in SHIFTS.items():
        shifted_text = apply_shift(doc_text, shift_name)
        shifted_emb = get_embedding(shifted_text)
        shift_vector = shifted_emb - original_emb if original_emb is not None and shifted_emb is not None else None
        
        result["shifts"][shift_name] = {
            "shift_vector": shift_vector,
            "shifted_text": shifted_text
        }
    
    return result

def calculate_average_shifts(documents_data: List[Dict]) -> Dict[str, np.ndarray]:
    """Calculate average shift vector for each shift type across all documents"""
    average_shifts = {}
    
    for shift_name in SHIFTS.keys():
        valid_vectors = []
        
        for doc_data in documents_data:
            shift_vector = doc_data["shifts"][shift_name]["shift_vector"]
            if shift_vector is not None:
                valid_vectors.append(shift_vector)
        
        if valid_vectors:
            average_shifts[shift_name] = np.mean(valid_vectors, axis=0)
        else:
            average_shifts[shift_name] = None
    
    return average_shifts

def get_isolated_shift(shift_vector: np.ndarray, average_shift: np.ndarray) -> np.ndarray:
    """Get isolated shift by subtracting average shift from document shift"""
    if shift_vector is None or average_shift is None:
        return None
    return shift_vector - average_shift

def save_shifted_texts(documents_data: List[Dict], documents: Dict[str, str]):
    """Save original and shifted texts to file for inspection"""
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(exist_ok=True)
    
    output_file = results_path / "shifted_texts.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_data in documents_data:
            doc_id = doc_data["doc_id"]
            original_text = documents.get(doc_id, "")
            
            f.write(f"\n{'='*80}\n")
            f.write(f"DOCUMENT: {doc_id}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("ORIGINAL TEXT:\n")
            f.write(f"{original_text}\n\n")
            
            for shift_name in SHIFTS.keys():
                shifted_text = doc_data["shifts"][shift_name].get("shifted_text", "")
                f.write(f"\n{'-'*80}\n")
                f.write(f"SHIFT: {shift_name}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{shifted_text}\n")
    
    print(f"Shifted texts saved to {output_file}")

def main():
    print("=" * 80)
    print("ShiftDim3 Ranking Test")
    print("=" * 80 + "\n")
    
    # Load documents
    print(f"Loading documents from '{SOURCE_DOCS_DIR}'...")
    documents = load_documents(SOURCE_DOCS_DIR)
    
    if not documents:
        print("No documents found!")
        return
    
    print(f"Loaded {len(documents)} documents\n")
    
    # PHASE 1: Process all documents ONCE in parallel
    print("=" * 80)
    print("PHASE 1: Processing Documents (Parallel)")
    print("=" * 80)
    
    documents_data = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_document, doc_id, doc_text): doc_id 
                  for doc_id, doc_text in documents.items()}
        
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                doc_data = future.result()
                documents_data.append(doc_data)
            except Exception as e:
                print(f"Error processing {doc_id}: {e}")
    
    print(f"\nSuccessfully processed {len(documents_data)} documents\n")
    
    # Save shifted texts for inspection
    save_shifted_texts(documents_data, documents)
    
    # Calculate average shifts across all documents
    print("=" * 80)
    print("Calculating Average Shifts Across Documents")
    print("=" * 80)
    average_shifts = calculate_average_shifts(documents_data)
    for shift_name in average_shifts:
        if average_shifts[shift_name] is not None:
            print(f"  {shift_name}: calculated")
    print()
    
    # PHASE 2: Process each query and rank against pre-computed documents
    print("=" * 80)
    print("PHASE 2: Processing Queries")
    print("=" * 80 + "\n")
    
    results = {}
    
    # Filter queries to only those referencing loaded documents
    loaded_doc_names = set(doc.replace('.txt', '') for doc in documents.keys())
    
    for i, query in enumerate(QUERY_MAPPINGS.keys(), 1):
        expected_doc = QUERY_MAPPINGS.get(query)
        
        # Skip query if expected doc is not in loaded documents
        if expected_doc not in loaded_doc_names:
            continue
        
        print(f"Query {i}: {query}")
        
        # Get query embeddings
        q_anchor = get_embedding(query)
        q_shifts = {}
        
        for shift_name, shift_prompt in SHIFTS.items():
            transformed = apply_shift(query, shift_name)
            q_shifted_emb = get_embedding(transformed)
            q_shifts[shift_name] = q_shifted_emb - q_anchor if q_anchor is not None and q_shifted_emb is not None else None
        
        # Score each document using pre-computed embeddings
        doc_scores = {}
        doc_shift_details = {}  # Track per-shift similarity for analysis
        
        for doc_data in documents_data:
            total_agreement = 0
            valid_shifts = 0
            shift_similarities = {}
            
            for shift_name in SHIFTS.keys():
                doc_shift = doc_data["shifts"][shift_name]["shift_vector"]
                query_shift = q_shifts[shift_name]
                avg_shift = average_shifts[shift_name]
                
                # Isolate shift by subtracting average shift
                isolated_doc_shift = get_isolated_shift(doc_shift, avg_shift)
                isolated_query_shift = get_isolated_shift(query_shift, avg_shift)
                
                if isolated_doc_shift is not None and isolated_query_shift is not None:
                    similarity = cosine_similarity(isolated_doc_shift, isolated_query_shift)
                    shift_similarities[shift_name] = similarity
                    total_agreement += similarity
                    valid_shifts += 1
            
            avg_score = total_agreement / valid_shifts if valid_shifts > 0 else 0
            doc_scores[doc_data["doc_id"]] = avg_score
            doc_shift_details[doc_data["doc_id"]] = shift_similarities
        
        # Sort by score
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get expected document
        expected_doc = QUERY_MAPPINGS.get(query)
        
        # Create result with ranking and correctness indicator
        result = []
        for doc_id, score in ranked:
            # Strip .txt extension for comparison
            doc_id_without_ext = doc_id.replace('.txt', '')
            is_correct = (doc_id_without_ext == expected_doc)
            result.append((doc_id, score, is_correct, doc_shift_details.get(doc_id, {})))
        
        results[query] = result
        print("  ✓ Ranked\n")
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")
    
    correct_count = 0
    
    for i, (query, ranked_docs) in enumerate(results.items(), 1):
        expected = QUERY_MAPPINGS.get(query)
        is_correct = ranked_docs[0][2] if ranked_docs else False
        
        if is_correct:
            correct_count += 1
        
        correct_indicator = "✓" if is_correct else "✗"
        
        print(f"Query {i}: {query}")
        print(f"Expected: {expected} {correct_indicator}\n")
        
        # Calculate separation percentage
        first_score = ranked_docs[0][1]
        second_score = ranked_docs[1][1] if len(ranked_docs) > 1 else 0
        
        if second_score > 0:
            separation_pct = ((first_score - second_score) / second_score) * 100
        else:
            separation_pct = 100 if first_score > 0 else 0
        
        print(f"1st: {ranked_docs[0][0]} ({ranked_docs[0][1]:.4f})")
        if len(ranked_docs) > 1:
            print(f"2nd: {ranked_docs[1][0]} ({ranked_docs[1][1]:.4f})")
            print(f"Separation: {separation_pct:.1f}%")
        else:
            print("2nd: N/A (only 1 document available)")
        
        # Show per-shift alignment and discriminatory power for the top result
        top_doc_shifts = ranked_docs[0][3]
        second_doc_shifts = ranked_docs[1][3] if len(ranked_docs) > 1 else {}
        
        if top_doc_shifts:
            print(f"\nShift Analysis (Alignment + Discriminatory Power):")
            sorted_shifts = sorted(top_doc_shifts.items(), key=lambda x: x[1], reverse=True)
            for shift_name, top_similarity in sorted_shifts:
                second_similarity = second_doc_shifts.get(shift_name, 0)
                discriminative_gap = top_similarity - second_similarity
                
                # Determine if it's both aligned AND discriminatory
                is_aligned = top_similarity > 0.5
                is_discriminatory = discriminative_gap > 0.1
                both = is_aligned and is_discriminatory
                
                if both:
                    indicator = "✓✓"  # Aligned AND discriminatory
                elif is_aligned:
                    indicator = "✓○"  # Aligned but not very discriminatory
                elif is_discriminatory:
                    indicator = "○✓"  # Discriminatory but not aligned
                else:
                    indicator = "✗✗"  # Neither aligned nor discriminatory
                
                print(f"  {indicator} {shift_name}: {top_similarity:.4f} (vs 2nd: {second_similarity:.4f}, gap: {discriminative_gap:.4f})")
        
        print()
    
    # Summary
    accuracy = (correct_count / len(results)) * 100 if results else 0
    print("=" * 80)
    print(f"ACCURACY: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    print("=" * 80)
    
    # Aggregate shift performance analysis
    print("\n" + "=" * 80)
    print("SHIFT PERFORMANCE AGGREGATION")
    print("=" * 80)
    
    shift_stats = {shift_name: {
        "aligned_count": 0,
        "discriminatory_count": 0,
        "both_count": 0,
        "total_similarity": 0,
        "total_gap": 0,
        "count": 0
    } for shift_name in SHIFTS.keys()}
    
    # Aggregate stats across all queries
    for query, ranked_docs in results.items():
        if ranked_docs:
            top_doc_shifts = ranked_docs[0][3]
            second_doc_shifts = ranked_docs[1][3] if len(ranked_docs) > 1 else {}
            
            for shift_name, top_similarity in top_doc_shifts.items():
                second_similarity = second_doc_shifts.get(shift_name, 0)
                discriminative_gap = top_similarity - second_similarity
                
                is_aligned = top_similarity > 0.5
                is_discriminatory = discriminative_gap > 0.1
                both = is_aligned and is_discriminatory
                
                shift_stats[shift_name]["total_similarity"] += top_similarity
                shift_stats[shift_name]["total_gap"] += discriminative_gap
                shift_stats[shift_name]["count"] += 1
                
                if is_aligned:
                    shift_stats[shift_name]["aligned_count"] += 1
                if is_discriminatory:
                    shift_stats[shift_name]["discriminatory_count"] += 1
                if both:
                    shift_stats[shift_name]["both_count"] += 1
    
    # Calculate averages and rank shifts
    shift_performance = []
    for shift_name, stats in shift_stats.items():
        if stats["count"] > 0:
            avg_similarity = stats["total_similarity"] / stats["count"]
            avg_gap = stats["total_gap"] / stats["count"]
            alignment_rate = (stats["aligned_count"] / stats["count"]) * 100
            discriminatory_rate = (stats["discriminatory_count"] / stats["count"]) * 100
            effectiveness_rate = (stats["both_count"] / stats["count"]) * 100
            
            shift_performance.append({
                "name": shift_name,
                "avg_similarity": avg_similarity,
                "avg_gap": avg_gap,
                "alignment_rate": alignment_rate,
                "discriminatory_rate": discriminatory_rate,
                "effectiveness_rate": effectiveness_rate,
                "count": stats["count"],
                "both_count": stats["both_count"],
                "aligned_count": stats["aligned_count"],
                "discriminatory_count": stats["discriminatory_count"]
            })
    
    # Sort by effectiveness (how often aligned AND discriminatory)
    shift_performance.sort(key=lambda x: x["effectiveness_rate"], reverse=True)
    
    print("\nShifts ranked by effectiveness (Aligned + Discriminatory):\n")
    print(f"{'Rank':<6} {'Shift Name':<25} {'Effectiveness':<15} {'Aligned':<10} {'Discrim.':<10} {'Avg Gap':<10}")
    print("-" * 80)
    
    for i, perf in enumerate(shift_performance, 1):
        print(f"{i:<6} {perf['name']:<25} {perf['effectiveness_rate']:>6.1f}% ({perf['both_count']}/{perf['count']:<3}) "
              f"{perf['alignment_rate']:>7.1f}% {perf['discriminatory_rate']:>8.1f}% {perf['avg_gap']:>9.4f}")
    
    print("\n" + "-" * 80)
    print("Legend:")
    print("  Effectiveness: % of queries where shift was both aligned (>0.5) and discriminatory (gap>0.1)")
    print("  Aligned: % of queries where top doc similarity >0.5")
    print("  Discrim.: % of queries where gap between 1st and 2nd doc >0.1")
    print("  Avg Gap: Average discriminative gap (1st - 2nd doc similarity)")
    print("=" * 80)

if __name__ == "__main__":
    main()
