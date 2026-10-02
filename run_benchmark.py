import json
import csv
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import ollama
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
EMBEDDING_MODEL = "nomic-embed-text"
MAX_CONTEXT_CHARS = 2000

BENCHMARK_FILES = [
    "LegalBench-RAG/benchmarks/cuad.json",
    "LegalBench-RAG/benchmarks/contractnli.json",
    "LegalBench-RAG/benchmarks/maud.json",
    "LegalBench-RAG/benchmarks/privacy_qa.json"
]
EMBEDDINGS_CSV = Path("embeddings/document_embeddings.csv")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding(text):
    """Get embedding for query text."""
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        if "embeddings" in response and len(response["embeddings"]) > 0:
            return np.array(response["embeddings"][0])
        return None
    except Exception as e:
        if "context length" not in str(e).lower():
            print(f"Error: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity."""
    if vec1 is None or vec2 is None:
        return 0
    try:
        return 1 - cosine(vec1, vec2)
    except:
        return 0

def load_document_embeddings():
    """Load document embeddings from CSV."""
    if not EMBEDDINGS_CSV.exists():
        print(f"❌ Embeddings CSV not found: {EMBEDDINGS_CSV}")
        return None
    
    embeddings = {}
    
    print(f"📂 Loading document embeddings...")
    with open(EMBEDDINGS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_name = row['document_name']
            # Extract embedding dimensions
            embedding = [float(row[f'dim_{i}']) for i in range(384)]
            embeddings[doc_name] = np.array(embedding)
    
    print(f"   Loaded {len(embeddings)} document embeddings")
    return embeddings

def find_most_similar_document(query, doc_embeddings):
    """Find the most similar document to the query."""
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return None, 0.0
    
    best_doc = None
    best_score = -1
    
    for doc_name, doc_embedding in doc_embeddings.items():
        score = cosine_similarity(query_embedding, doc_embedding)
        if score > best_score:
            best_score = score
            best_doc = doc_name
    
    return best_doc, best_score

def normalize_path(path_str):
    """Normalize file path for comparison."""
    # Replace all slashes and backslashes with underscores
    normalized = path_str.replace("\\", "_").replace("/", "_")
    return normalized.lower()

def find_matching_document(benchmark_path, doc_embeddings):
    """Find document that matches the benchmark path."""
    # Normalize the benchmark path
    normalized_bench = normalize_path(benchmark_path)
    
    # Try exact match first
    for doc_name in doc_embeddings.keys():
        if normalize_path(doc_name) == normalized_bench:
            return doc_name
    
    # Try partial match (last part of path)
    bench_filename = Path(benchmark_path).name.lower()
    for doc_name in doc_embeddings.keys():
        if doc_name.lower() == bench_filename or \
           doc_name.lower().endswith(bench_filename) or \
           normalize_path(doc_name).endswith(normalized_bench):
            return doc_name
    
    return None

def run_benchmark():
    """Run benchmark on all test queries."""
    
    # Load embeddings
    doc_embeddings = load_document_embeddings()
    if doc_embeddings is None:
        return False
    
    print("\n" + "=" * 70)
    print("🧪 RUNNING BENCHMARK")
    print("=" * 70)
    
    total_queries = 0
    correct_retrieval = 0
    
    # Load and process each benchmark file
    for benchmark_file in BENCHMARK_FILES:
        benchmark_path = Path(benchmark_file)
        
        if not benchmark_path.exists():
            print(f"⚠️  Skipping {benchmark_file} (not found)")
            continue
        
        print(f"\n📋 Processing {benchmark_path.name}...")
        
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tests = data.get('tests', [])
        file_correct = 0
        
        with tqdm(total=len(tests), desc="  Testing", unit="query") as pbar:
            for test in tests:
                if not test.get('snippets'):
                    pbar.update(1)
                    continue
                
                query = test.get('query', '')
                
                # Get ground truth documents
                ground_truth_docs = set()
                for snippet in test['snippets']:
                    file_path = snippet.get('file_path', '')
                    if file_path:
                        # Find matching document for this path
                        matching_doc = find_matching_document(file_path, doc_embeddings)
                        if matching_doc:
                            ground_truth_docs.add(matching_doc)
                
                if not ground_truth_docs or not query.strip():
                    pbar.update(1)
                    continue
                
                # Find most similar document
                retrieved_doc, similarity = find_most_similar_document(query, doc_embeddings)
                
                # Check if retrieved document is in ground truth
                if retrieved_doc and retrieved_doc in ground_truth_docs:
                    correct_retrieval += 1
                    file_correct += 1
                
                total_queries += 1
                
                # Update progress with current accuracy
                current_accuracy = (file_correct / max(total_queries, 1) * 100)
                pbar.set_postfix({'correct': f"{file_correct}/{total_queries}", 'accuracy': f"{current_accuracy:.1f}%"})
                pbar.update(1)
        
        accuracy = (file_correct / len(tests) * 100) if tests else 0
        print(f"  Accuracy: {accuracy:.1f}% ({file_correct}/{len(tests)})")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    overall_accuracy = (correct_retrieval / total_queries * 100) if total_queries > 0 else 0
    print(f"\n  Total queries: {total_queries}")
    print(f"  Correct retrievals: {correct_retrieval}")
    print(f"  Overall accuracy: {overall_accuracy:.2f}%")
    print("=" * 70 + "\n")
    
    return True

if __name__ == "__main__":
    success = run_benchmark()
    exit(0 if success else 1)
