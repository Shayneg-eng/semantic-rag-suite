import os
import json
import numpy as np
import openai
import ollama
from typing import List, Dict, Any

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

POE_API_KEY = os.getenv("POE_API_KEY", "")

SHIFT_PROMPTS = {
    "shift_1_entity": "Extract only the subject-verb-object relationships. Strip all adjectives and context. Format as simple declarative sentences.",
    "shift_2_abstraction": "Rewrite this using higher-level abstract concepts. (e.g., 'iPhone' -> 'mobile device', 'login' -> 'authentication').",
    "shift_3_certainty": "Remove all hedging (maybe, possibly, typically). Rewrite every sentence as an absolute, 100% certain fact.",
    "shift_4_causal": "Remove all temporal and causal connectors (because, therefore, then). List the information as independent atomic facts.",
    "shift_5_passive": "Rewrite the entire text in the passive voice. Remove all mentions of who is doing the action.",
    "shift_6_negation": "Flip the polarity. Rewrite positive assertions as negative ones, and negative ones as positive.",
    "shift_7_density": "Summarize this into the shortest possible version that retains all unique information (maximum compression).",
    "shift_8_assumption": "Identify and list only the unstated assumptions required for this text to be true."
}

poe_client = openai.OpenAI(api_key=POE_API_KEY, base_url="https://api.poe.com/v1")

# ==============================================================================
# 2. CORE FUNCTIONS
# ==============================================================================

def get_embedding(text: str) -> np.ndarray:
    """Generates vector using Nomic Embed Text (local)"""
    if not text or not text.strip():
        return np.zeros(768)
    response = ollama.embed(
        model='nomic-embed-text',
        input=text
    )
    return np.array(response['embeddings'][0])

def generate_shift(text: str, instruction: str) -> str:
    try:
        response = poe_client.chat.completions.create(
            model="llama-3.1-8b-cs",
            messages=[{"role": "user", "content": f"{instruction}\n\nTEXT:\n{text}"}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [!] Generation failed: {e}")
        return text

def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

# ==============================================================================
# 4. THE DIFFERENTIAL RETRIEVER (Pure Directional Logic)
# ==============================================================================

def query_differential_rag(query: str, index_file: str):
    print(f"\n🔍 Querying (Noise-Cancelled Fingerprint V4): '{query}'")
    
    with open(index_file, "r") as f:
        index = json.load(f)

    # 1. Compute Global Mean Deltas for Noise Cancellation
    # This identifies the "Common Mode" (e.g., what 'Passive Voice' looks like on average)
    mean_deltas = {name: [] for name in SHIFT_PROMPTS.keys()}
    for doc_data in index.values():
        d_anchor = np.array(doc_data["anchor_vector"])
        for s_name, s_data in doc_data["shifts"].items():
            mean_deltas[s_name].append(np.array(s_data["vector"]) - d_anchor)
    
    # Calculate the average 'Style Vector' for each shift
    noise_vectors = {s_name: np.mean(deltas, axis=0) for s_name, deltas in mean_deltas.items()}

    # 2. Process Query and Subtract Noise
    q_anchor_vec = get_embedding(query)
    q_deltas = {}
    for shift_name, prompt in SHIFT_PROMPTS.items():
        s_vec = get_embedding(generate_shift(query, prompt))
        raw_delta = s_vec - q_anchor_vec
        # NOISE CANCELLATION: Remove the average 'style' movement
        q_deltas[shift_name] = raw_delta - noise_vectors[shift_name]

    # 3. Score Documents with Noise-Cancelled Deltas
    results = []
    for doc_name, doc_data in index.items():
        d_anchor = np.array(doc_data["anchor_vector"])
        alignments = []
        
        for s_name, q_delta_clean in q_deltas.items():
            if s_name in doc_data["shifts"]:
                raw_d_delta = np.array(doc_data["shifts"][s_name]["vector"]) - d_anchor
                # Subtract the same noise from the document delta
                d_delta_clean = raw_d_delta - noise_vectors[s_name]
                
                alignments.append(cosine_similarity(q_delta_clean, d_delta_clean))
        
        fingerprint = np.mean(alignments) if alignments else 0
        base_rag = cosine_similarity(q_anchor_vec, d_anchor)
        
        # Boost the signal: Multiply Base by the Fingerprint
        # This rewards docs that are topically close AND structurally identical
        final_score = base_rag * (1 + fingerprint)
        
        results.append((doc_name, final_score, fingerprint, base_rag))

    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 RESULTS (Noise-Cancelled Semantic Fingerprint):")
    print(f"{'DOC NAME':<15} | {'FINAL':<8} | {'CLEAN DELTA':<12} | {'BASE RAG':<8}")
    print("-" * 65)
    for name, final, clean, base in results[:10]:
        print(f"{name:<15} | {final:.4f} | {clean:.4f}       | {base:.4f}")

if __name__ == "__main__":
    if os.path.exists("rag_index.json"):
        query_differential_rag("How do I use tokens for secure authorization?", "rag_index.json")
    else:
        print("Please run build_index() first.")