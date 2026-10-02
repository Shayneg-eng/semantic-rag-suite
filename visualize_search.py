import json
import csv
import numpy as np
from ollama import embed
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

# Load document embeddings
embeddings_file = "embeddings/document_embeddings.csv"
doc_embeddings = {}
doc_paths = {}
doc_names_list = []

print("Loading document embeddings...")
with open(embeddings_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        doc_name = row['document_name']
        doc_path = row['document_path']
        
        # Extract the embedding dimensions (768 dimensions)
        embedding = np.array([float(row[f'dim_{i}']) for i in range(768)])
        
        doc_embeddings[doc_name] = embedding
        doc_paths[doc_name] = doc_path
        doc_names_list.append(doc_name)

print(f"Loaded {len(doc_embeddings)} documents\n")

def get_embedding(text):
    """Get embedding for text using ollama"""
    response = embed(model="nomic-embed-text", input=text)
    return np.array(response['embeddings'][0])

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def normalize_doc_name(path):
    """Normalize document path to match embeddings naming"""
    normalized = path.replace('/', '_').replace('\\', '_')
    return normalized

def visualize_query(query_text, correct_file_path, save_file=None):
    """Create visualization of similarity scores for a query"""
    
    print(f"Query: {query_text[:100]}...")
    print(f"Expected: {correct_file_path}")
    
    # Embed the query
    query_embedding = get_embedding(query_text)
    
    # Calculate similarity to all documents
    similarities = []
    correct_normalized = normalize_doc_name(correct_file_path).lower()
    correct_idx = -1
    
    for idx, doc_name in enumerate(doc_names_list):
        doc_embedding = doc_embeddings[doc_name]
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((idx, doc_name, sim))
        
        # Check if this is the correct document
        doc_normalized = doc_name.lower()
        if doc_normalized == correct_normalized or \
           correct_file_path.replace('/', '_').lower() == doc_normalized or \
           correct_file_path.replace('\\', '_').lower() == doc_normalized or \
           os.path.basename(correct_file_path).lower() in doc_normalized:
            correct_idx = idx
    
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: All documents sorted by similarity
    indices = [s[0] for s in similarities]
    sim_scores = [s[2] for s in similarities]
    colors = ['green' if s[0] == correct_idx else 'steelblue' for s in similarities]
    
    ax1.bar(range(len(similarities)), sim_scores, color=colors, alpha=0.7, width=1.0)
    ax1.set_xlabel('Documents (sorted by similarity)', fontsize=12)
    ax1.set_ylabel('Similarity Score', fontsize=12)
    ax1.set_title(f'All Document Similarities (Correct doc in green)\n{query_text[:80]}...', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Find the rank of correct document
    correct_rank = -1
    for rank, (idx, name, sim) in enumerate(similarities, 1):
        if idx == correct_idx:
            correct_rank = rank
            break
    
    if correct_rank > 0:
        # Add annotation
        ax1.text(0.02, 0.95, f'✅ Correct at rank #{correct_rank}', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    else:
        ax1.text(0.02, 0.95, f'❌ Correct document not found', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                verticalalignment='top')
    
    # Plot 2: Top 30 documents
    top_30 = similarities[:30]
    top_names = [f"{s[1][:40]}..." if len(s[1]) > 40 else s[1] for s in top_30]
    top_sims = [s[2] for s in top_30]
    top_colors = ['green' if s[0] == correct_idx else 'steelblue' for s in top_30]
    
    ax2.barh(range(len(top_30)), top_sims, color=top_colors, alpha=0.7)
    ax2.set_yticks(range(len(top_30)))
    ax2.set_yticklabels(top_names, fontsize=8)
    ax2.set_xlabel('Similarity Score', fontsize=12)
    ax2.set_title('Top 30 Most Similar Documents', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    ax2.set_xlim([0, 1.0])
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to {save_file}\n")
    
    plt.show()
    
    return correct_rank

# Load a sample benchmark
benchmark_file = "LegalBench-RAG/benchmarks/contractnli.json"

if not os.path.exists(benchmark_file):
    print(f"Benchmark file not found: {benchmark_file}")
    sys.exit(1)

with open(benchmark_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

tests = data.get('tests', [])
print(f"Total queries in benchmark: {len(tests)}\n")

# Process first 5 queries
for i, test in enumerate(tests[:5]):
    query = test.get('query', '')
    snippets = test.get('snippets', [])
    
    if not snippets:
        continue
    
    correct_file = snippets[0].get('file_path', '')
    
    # Create a safe filename for saving
    safe_name = f"query_{i+1}".replace(' ', '_')
    save_path = f"visualizations/{safe_name}.png"
    os.makedirs("visualizations", exist_ok=True)
    
    visualize_query(query, correct_file, save_path)
    
    print("=" * 80)
