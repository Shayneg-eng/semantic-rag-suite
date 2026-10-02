import os
import json
import numpy as np
import openai
import ollama
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

POE_API_KEY = os.getenv("POE_API_KEY", "")

# Updated shift prompts matching testing3.py
SHIFT_PROMPTS = {
    "shift_1_entity": "Extract only the subject-verb-object relationships. Strip all adjectives and context. Format as simple declarative sentences.",
    "shift_2_abstraction": "Rewrite this using higher-level abstract concepts. (e.g., 'iPhone' -> 'mobile device', 'login' -> 'authentication').",
    "shift_3_certainty": "Remove all hedging (maybe, possibly, typically). Rewrite every sentence as an absolute, 100% certain fact.",
    "shift_5_passive": "Rewrite the entire text in the passive voice. Remove all mentions of who is doing the action.",
    "shift_6_negation": "Flip the polarity. Rewrite positive assertions as negative ones, and negative ones as positive.",
    "shift_7_density": "Summarize this into the shortest possible version that retains all unique information (maximum compression).",
    "shift_9_perspective": "Rewrite from the opposite stakeholder's perspective. If it's about a provider, rewrite as if from a consumer's view, and vice versa.",
    "shift_10_temporal": "Rewrite this with all temporal markers removed. Convert all tenses to present tense and remove time references (yesterday, next year, etc).",
    "shift_11_modality": "Rewrite replacing all modal verbs (could, should, might) with definitive statements. Convert all possibilities to actualities.",
    "shift_12_agent_flip": "Swap all active agents with their recipients or counterparts. If X does Y to Z, rewrite as Z does Y to X.",
    "shift_13_scope": "Zoom in on micro-level details and granular components. Break down abstract concepts into their concrete parts.",
    "shift_14_causality_invert": "Invert all cause-effect relationships. Rewrite as if effects are causes and causes are effects.",
    "shift_15_contrast": "Extract only contrasts, opposites, and comparative relationships. Emphasize what differs rather than similarities.",
    "shift_16_exemplification": "Convert all generic statements to specific examples, and all examples to generic patterns they represent."
}

poe_client = openai.OpenAI(api_key=POE_API_KEY, base_url="https://api.poe.com/v1")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text using nomic-embed-text"""
    if not text or not text.strip():
        return np.zeros(768)
    try:
        response = ollama.embed(model='nomic-embed-text', input=text)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"    [ERROR] Embedding failed: {e}")
        return np.zeros(768)

def generate_shift(text: str, instruction: str) -> str:
    """Generate a shifted version of text using LLM"""
    try:
        response = poe_client.chat.completions.create(
            model="llama-3.1-8b-cs",
            messages=[{"role": "user", "content": f"{instruction}\n\nTEXT:\n{text}"}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [ERROR] Shift generation failed: {e}")
        return text

# ==============================================================================
# INDEX GENERATION
# ==============================================================================

def generate_index(source_docs_dir: str = "source_docs", output_file: str = "rag_index.json"):
    """Generate RAG index with all shifts"""
    
    print(f"Generating RAG index from {source_docs_dir}/...")
    print(f"Using {len(SHIFT_PROMPTS)} shifts")
    print()
    
    index = {}
    
    # Get all source documents
    source_path = Path(source_docs_dir)
    doc_files = sorted([f for f in source_path.glob("*.txt")])
    
    total_docs = len(doc_files)
    
    for doc_idx, doc_file in enumerate(doc_files, 1):
        doc_name = doc_file.stem
        print(f"[{doc_idx}/{total_docs}] Processing {doc_name}...")
        
        # Read document
        doc_text = doc_file.read_text()
        
        # Get anchor embedding (base document)
        print(f"  [1/{len(SHIFT_PROMPTS)+1}] Computing anchor embedding...")
        anchor_vec = get_embedding(doc_text)
        
        # Generate embeddings for all shifts
        shifts_data = {}
        for shift_idx, (shift_name, prompt) in enumerate(SHIFT_PROMPTS.items(), 1):
            print(f"  [{shift_idx+1}/{len(SHIFT_PROMPTS)+1}] {shift_name}...")
            
            # Generate shifted text
            shifted_text = generate_shift(doc_text, prompt)
            
            # Get embedding for shifted text
            shifted_vec = get_embedding(shifted_text)
            
            shifts_data[shift_name] = {
                "vector": shifted_vec.tolist(),
                "text_preview": shifted_text[:200]  # Store preview for debugging
            }
        
        # Store in index
        index[doc_name] = {
            "anchor_vector": anchor_vec.tolist(),
            "shifts": shifts_data
        }
        
        print()
    
    # Save index
    print(f"Saving index to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✅ Index generated successfully!")
    print(f"   Documents: {len(index)}")
    print(f"   Shifts per document: {len(SHIFT_PROMPTS)}")
    print(f"   Total embeddings: {len(index) * (len(SHIFT_PROMPTS) + 1)}")

if __name__ == "__main__":
    generate_index()
