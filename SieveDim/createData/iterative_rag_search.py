import os
import sys
import pickle
import numpy as np
import openai
from pathlib import Path

# Configuration
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA', 'vector_db.pkl')
EMBEDDING_MODEL = 'nomic-embed-text'
POE_API_KEY = os.getenv("POE_API_KEY", "")
MAX_ITERATIONS = 20

# Setup OpenAI client for Poe
client = openai.OpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

sys.path.insert(0, os.path.dirname(__file__))
from search_and_rank import get_query_embedding

def cosine_distance(a, b):
    """Compute cosine distance between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1 - (np.dot(a, b) / (norm_a * norm_b))

def rag_search(query_embedding, all_doc_embeddings, doc_names_list, top_k=5):
    """Perform RAG search and return top-k results."""
    query_embedding = np.array(query_embedding, dtype=np.float32)
    all_doc_embeddings = np.array(all_doc_embeddings, dtype=np.float32)
    
    # Compute distances
    query_norm = np.linalg.norm(query_embedding)
    dot_products = all_doc_embeddings @ query_embedding
    doc_norms = np.linalg.norm(all_doc_embeddings, axis=1)
    
    valid = doc_norms > 0
    distances = np.ones(len(doc_norms))
    distances[valid] = 1 - (dot_products[valid] / (doc_norms[valid] * query_norm))
    
    # Get top-k
    sorted_indices = np.argsort(distances)
    top_docs = [(doc_names_list[i], 1 - distances[i]) for i in sorted_indices[:top_k]]
    
    return top_docs

def refine_query_with_llm(original_query, current_query, top_result_name, iteration, search_history):
    """Use LLM to refine the search query and evaluate if we've found a good answer.
    Returns: (new_query, confidence_score, should_stop)
    """
    
    # Format search history for LLM context
    history_str = "\n".join([
        f"  Iteration {h['iteration']}: '{h['query']}' → {h['top_doc'][:60]} (sim: {h['similarity']:.3f})"
        for h in search_history
    ])
    
    prompt = f"""You are an intelligent search query refinement assistant with access to search history.

ORIGINAL USER INTENT: "{original_query}"
CURRENT ITERATION: {iteration}/10

SEARCH HISTORY (what we've tried so far):
{history_str}

LAST SEARCH RESULT:
Query tried: "{current_query}"
Top document returned: {top_result_name}

YOUR TASK - Provide TWO evaluations:

1. CONFIDENCE SCORE (0-100): How confident are you that the current top result matches the original user intent? 
   - 0-30: Definitely wrong document
   - 31-60: Possibly relevant but uncertain
   - 61-85: Likely a good match
   - 86-100: Definitely correct/excellent match

2. NEXT ACTION: 
   If confidence ≥ 85, return: "STOP|[confidence]"
   If confidence < 85 AND iteration < 10, return: "CONTINUE|[confidence]|[new_query]"
   If iteration = 10, return: "STOP|[confidence]"

For the new query (if continuing):
- Review the history to see what has NOT been tried
- Avoid repeating failed approaches
- Try a different angle/perspective related to "{original_query}"
- Make it significantly different from all previous queries
- Keep it concise (5-10 words)

RESPOND IN THIS EXACT FORMAT:
SCORE|[number]|ACTION|[STOP/CONTINUE]|[new_query if continuing, else empty]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-cs",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse response
        parts = result.split("|")
        if len(parts) >= 3:
            try:
                confidence = int(parts[1])
                action = parts[2].strip().upper()
                new_query = parts[3].strip() if len(parts) > 3 else current_query
                should_stop = action == "STOP" or confidence >= 85
                return new_query, confidence, should_stop
            except:
                pass
        
        # Fallback
        return current_query, 50, False
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return current_query, 50, False

def iterative_rag_search(user_query, vector_db, doc_names_list, doc_embeddings_array):
    """Perform iterative RAG search with LLM query refinement and early stopping."""
    
    print("\n" + "=" * 80)
    print("ITERATIVE RAG SEARCH WITH EARLY STOPPING & HISTORY AWARENESS")
    print("=" * 80)
    print(f"\nOriginal Query: {user_query}\n")
    
    current_query = user_query
    search_history = []
    last_similarities = []
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[Iteration {iteration}/{MAX_ITERATIONS}]")
        print(f"Searching with query: \"{current_query}\"")
        
        # Get embedding for current query
        try:
            query_embedding = get_query_embedding(current_query)
        except Exception as e:
            print(f"Error embedding query: {e}")
            break
        
        # Perform RAG search
        top_results = rag_search(query_embedding, doc_embeddings_array, doc_names_list, top_k=1)
        
        if not top_results:
            print("No results found!")
            break
        
        top_doc, similarity = top_results[0]
        print(f"  Top Result: {top_doc[:70]}")
        print(f"  Similarity: {similarity:.4f}")
        
        search_history.append({
            'iteration': iteration,
            'query': current_query,
            'top_doc': top_doc,
            'similarity': similarity
        })
        
        last_similarities.append(similarity)
        
        # Check if this is the last iteration
        if iteration == MAX_ITERATIONS:
            print(f"\n✓ Reached maximum iterations!")
            break
        
        # Evaluate with LLM (includes confidence scoring and early stopping logic)
        print(f"  Evaluating with LLM (checking if answer found)...")
        new_query, confidence, should_stop = refine_query_with_llm(
            user_query, current_query, top_doc, iteration, search_history
        )
        
        print(f"  LLM Confidence: {confidence}/100")
        
        if should_stop:
            print(f"  ✓ STOPPING EARLY: Found good match!")
            break
        
        # Check for convergence (similarity not improving for 2+ iterations)
        if len(last_similarities) >= 3:
            recent_sims = last_similarities[-3:]
            if recent_sims[0] >= recent_sims[1] >= recent_sims[2]:
                # Similarity plateauing or decreasing
                if abs(recent_sims[0] - recent_sims[2]) < 0.01:
                    print(f"  ⚠ Converged (no improvement in last 3 iterations)")
                    if confidence >= 60:
                        print(f"  ✓ STOPPING: Converged with acceptable confidence")
                        break
        
        current_query = new_query
        print(f"  New query generated: \"{current_query}\"")
    
    return search_history
    
    return search_history

def main():
    """Main function for iterative RAG search."""
    
    # Load vector database
    print("Loading vector database...")
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"Error: Vector database not found at {VECTOR_DB_PATH}")
        return
    
    with open(VECTOR_DB_PATH, 'rb') as f:
        vector_db = pickle.load(f)
    
    doc_names_list = sorted(list(vector_db.keys()))
    doc_embeddings_array = np.array([np.array(vector_db[name]['embedding'], dtype=np.float32) 
                                     for name in doc_names_list], dtype=np.float32)
    
    print(f"✓ Loaded {len(vector_db)} document embeddings\n")
    
    # Get user input
    user_query = input("Enter your search query: ").strip()
    if not user_query:
        print("Query cannot be empty.")
        return
    
    # Perform iterative search
    history = iterative_rag_search(user_query, vector_db, doc_names_list, doc_embeddings_array)
    
    # Display final results
    print("\n" + "=" * 80)
    print("SEARCH HISTORY SUMMARY")
    print("=" * 80)
    
    print(f"\nIteration | Query | Top Document | Similarity")
    print("-" * 80)
    
    best_result = None
    best_similarity = -1
    
    for h in history:
        iteration = h['iteration']
        query = h['query'][:30]
        doc = h['top_doc'][:45]
        sim = h['similarity']
        
        print(f"{iteration:9d} | {query:30s} | {doc:45s} | {sim:.4f}")
        
        if sim > best_similarity:
            best_similarity = sim
            best_result = h
    
    print("-" * 80)
    print(f"\n✓ Best Result Found at Iteration {best_result['iteration']}:")
    print(f"  Query: {best_result['query']}")
    print(f"  Document: {best_result['top_doc']}")
    print(f"  Similarity: {best_result['similarity']:.4f}")
    
    # Load and display the document content
    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'corpus_chunks', best_result['top_doc'])
    
    print("\n" + "=" * 80)
    print("FINAL DOCUMENT CONTENT")
    print("=" * 80)
    
    if os.path.exists(corpus_path):
        try:
            with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Display first 1500 characters and indicate if truncated
            if len(content) > 1500:
                print(f"\n{content[:1500]}\n\n... [Document truncated - {len(content)} total characters]")
            else:
                print(f"\n{content}")
        except Exception as e:
            print(f"Error reading document: {e}")
    else:
        print(f"Error: Document file not found at {corpus_path}")

if __name__ == '__main__':
    main()
