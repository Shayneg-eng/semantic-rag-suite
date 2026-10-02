import os
import json
import numpy as np
import openai
import ollama
from typing import List, Dict, Any

# ==============================================================================
# 1. CONFIGURATION & TEST BANK
# ==============================================================================

POE_API_KEY = os.getenv("POE_API_KEY", "")

# Ground Truth Test Bank (2 prompts per document)
TEST_SUITE = {
    "DOC_1.txt": [
        "How do I implement secure third-party access to my API using tokens?",
        "What are the best practices for managing access token expiration and refresh in OAuth2?"
    ],
    "DOC_2.txt": [
        "How do composite indexes improve database query performance?",
        "What is the difference between B-tree and hash indexes in SQL databases?"
    ],
    "DOC_3.txt": [
        "How do I decide between synchronous and asynchronous communication in microservices?",
        "What are the advantages of using a service registry for discovery in a distributed system?"
    ],
    "DOC_4.txt": [
        "What is the cache-aside pattern and how does it handle cache misses?",
        "How does Redis replication and sentinel ensure high availability during failure?"
    ],
    "DOC_5.txt": [
        "How can I reconstruct a request flow across distributed services using logs and correlation IDs?",
        "What are the differences between log levels like INFO, WARN, and ERROR in production?"
    ],
    "DOC_6.txt": [
        "What is the principle of least privilege and how does it apply to account security?",
        "How do parameterized queries protect against SQL injection attacks?"
    ],
    "DOC_7.txt": [
        "What are the differences between Layer 4 and Layer 7 load balancing?",
        "How does weighted round-robin distribution handle servers with different capacities?"
    ],
    "DOC_8.txt": [
        "What are the trade-offs between synchronous and asynchronous database replication?",
        "How do consensus algorithms like Raft manage automatic failover in distributed databases?"
    ],
    "DOC_9.txt": [
        "What is the difference between token bucket and leaky bucket rate limiting algorithms?",
        "How should clients implement backoff strategies when hitting HTTP 429 status codes?"
    ],
    "DOC_10.txt": [
        "How does the circuit breaker pattern protect services from cascading failures?",
        "What is exponential backoff with jitter and why is it used for network retries?"
    ]
}

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
# 2. CORE ENGINE FUNCTIONS
# ==============================================================================

def get_embedding(text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(768)
    response = ollama.embed(model='nomic-embed-text', input=text)
    return np.array(response['embeddings'][0])

def generate_shift(text: str, instruction: str) -> str:
    try:
        response = poe_client.chat.completions.create(
            model="llama-3.1-8b-cs",
            messages=[{"role": "user", "content": f"{instruction}\n\nTEXT:\n{text}"}]
        )
        return response.choices[0].message.content.strip()
    except:
        return text

def cosine_similarity(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return np.dot(v1, v2) / (n1 * n2) if n1 > 0 and n2 > 0 else 0

# ==============================================================================
# 3. BENCHMARKING LOGIC
# ==============================================================================

def run_benchmark(index_file: str):
    with open(index_file, "r") as f:
        index = json.load(f)

    # Pre-calculate Global Noise Vectors
    mean_deltas = {name: [] for name in SHIFT_PROMPTS.keys()}
    for doc_data in index.values():
        d_anchor = np.array(doc_data["anchor_vector"])
        for s_name, s_data in doc_data["shifts"].items():
            mean_deltas[s_name].append(np.array(s_data["vector"]) - d_anchor)
    noise_vectors = {s_name: np.mean(deltas, axis=0) for s_name, deltas in mean_deltas.items()}

    metrics = {
        "standard_rag": {"accuracy": 0, "mrr": 0, "gap": []},
        "fingerprint_v4": {"accuracy": 0, "mrr": 0, "gap": []}
    }
    
    total_queries = 0

    for target_doc, queries in TEST_SUITE.items():
        for query in queries:
            total_queries += 1
            print(f"\n[{total_queries}] Testing Query for {target_doc}: '{query[:50]}...'")
            
            # --- PROCESS QUERY ---
            q_anchor_vec = get_embedding(query)
            q_deltas_clean = {}
            for s_name, prompt in SHIFT_PROMPTS.items():
                s_vec = get_embedding(generate_shift(query, prompt))
                q_deltas_clean[s_name] = (s_vec - q_anchor_vec) - noise_vectors[s_name]

            # --- SCORE ALL DOCS ---
            rag_results = []
            fingerprint_results = []

            for doc_name, doc_data in index.items():
                d_anchor = np.array(doc_data["anchor_vector"])
                
                # Standard RAG Score
                base_score = cosine_similarity(q_anchor_vec, d_anchor)
                rag_results.append((doc_name, base_score))

                # Fingerprint Score
                alignments = []
                for s_name, q_delta_clean in q_deltas_clean.items():
                    if s_name in doc_data["shifts"]:
                        d_delta_clean = (np.array(doc_data["shifts"][s_name]["vector"]) - d_anchor) - noise_vectors[s_name]
                        alignments.append(cosine_similarity(q_delta_clean, d_delta_clean))
                
                f_score = base_score * (1 + np.mean(alignments)) if alignments else base_score
                fingerprint_results.append((doc_name, f_score))

            # --- EVALUATE METRICS ---
            for method, results, key in [("Standard RAG", rag_results, "standard_rag"), 
                                         ("Fingerprint V4", fingerprint_results, "fingerprint_v4")]:
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Accuracy & Rank
                rank = next(i for i, (name, _) in enumerate(results) if name == target_doc) + 1
                if rank == 1: metrics[key]["accuracy"] += 1
                metrics[key]["mrr"] += (1.0 / rank)
                
                # Confidence Gap (Score 1 - Score 2)
                gap = results[0][1] - results[1][1]
                metrics[key]["gap"].append(gap)

                print(f"  {method}: Rank {rank} | Score: {results[0][1]:.4f} | Gap: {gap:.4f}")

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print("FINAL PERFORMANCE METRICS")
    print("="*60)
    for key, name in [("standard_rag", "Standard RAG"), ("fingerprint_v4", "Fingerprint V4")]:
        acc = (metrics[key]["accuracy"] / total_queries) * 100
        mrr = metrics[key]["mrr"] / total_queries
        avg_gap = np.mean(metrics[key]["gap"])
        print(f"{name}:")
        print(f"  - Top-1 Accuracy: {acc:.1f}%")
        print(f"  - MRR:            {mrr:.4f}")
        print(f"  - Avg. Winning Gap: {avg_gap:.4f} (System Confidence)")
        print("-" * 30)

if __name__ == "__main__":
    if os.path.exists("rag_index.json"):
        run_benchmark("rag_index.json")
    else:
        print("Please ensure rag_index.json is in the current directory.")