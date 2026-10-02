import os
from pathlib import Path
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import tiktoken

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

# Use cl100k_base encoding (closest to DeepSeek's tokenizer)
try:
    encoding = tiktoken.get_encoding("cl100k_base")
except:
    # Fallback: estimate 1 token per 4 characters
    encoding = None

def count_tokens(text):
    """Count tokens in text."""
    if encoding:
        return len(encoding.encode(text))
    else:
        # Rough estimate: 1 token per 4 characters
        return len(text) // 4

def create_embedding_summary(document_content, doc_name, attempt=1):
    """Create a concise, embedding-friendly summary of a document.
    Regenerates if token count exceeds 1500."""
    
    MAX_TOKEN_LIMIT = 1500
    MAX_ATTEMPTS = 3
    
    # Truncate very large documents to first 200000 chars to avoid token limits
    if len(document_content) > 200000:
        content_sample = document_content[:200000]
    else:
        content_sample = document_content
    
    # Adjust prompt based on attempt
    if attempt == 1:
        brevity_instruction = "MUST be under 1500 tokens."
    else:
        brevity_instruction = f"MUST be MUCH MORE CONCISE - attempt {attempt}/3. Target: under 1200 tokens. Remove less critical details."
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"""You are a legal document summarizer specialized in creating concise, embedding-optimized summaries.

Your summaries MUST be:
- {brevity_instruction}
- Structured with clear key-value pairs for optimal embedding
- Rich with specific legal terminology and concepts
- Factual and direct

Format your summary as:

DOCUMENT_TYPE: [e.g., Non-Disclosure Agreement, Service Contract, etc.]
PARTIES: [who are the main parties/roles]
KEY_TERMS: [3-5 main concepts separated by commas]
DURATION: [length of agreement if stated]
JURISDICTION: [location/law if mentioned]
MAIN_CLAUSES: [2-3 main obligation clauses]
SPECIAL_PROVISIONS: [any unique provisions, requirements, or notable terms]
RESTRICTIONS: [any major restrictions or limitations]

Then provide a 2-3 sentence summary capturing the essence.

Focus on: specificity, legal concepts, party roles, obligations, restrictions, and unique provisions. This will help embeddings find relevant documents."""},
            {"role": "user", "content": f"""Summarize this document for embedding search:

{content_sample}"""}
        ],
        stream=False
    )
    
    summary = response.choices[0].message.content.strip()
    token_count = count_tokens(summary)
    
    # Check if token count is acceptable
    if token_count <= MAX_TOKEN_LIMIT:
        return summary, token_count
    elif attempt < MAX_ATTEMPTS:
        # Regenerate with stricter brevity requirement
        return create_embedding_summary(document_content, doc_name, attempt=attempt+1)
    else:
        # Max attempts reached, return truncated version
        # Remove last 25% of content
        truncated = summary[:int(len(summary) * 0.75)]
        new_token_count = count_tokens(truncated)
        return truncated, new_token_count

def summarize_document_batch(doc_path):
    """Process a single document and return (success, doc_name, summary, token_count or error)."""
    try:
        doc_name = doc_path.name
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        summary, token_count = create_embedding_summary(content, doc_name)
        return True, doc_name, summary, token_count
    except Exception as e:
        return False, doc_path.name, str(e), 0

def save_summary(doc_name, summary, token_count, output_folder):
    """Save summary to output folder with matching name and token count."""
    # Replace .txt with _summary.txt
    summary_name = doc_name.replace('.txt', '_summary.txt')
    summary_path = Path(output_folder) / summary_name
    
    # Prepend token count metadata
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"[TOKENS: {token_count}]\n\n")
        f.write(summary)
    
    return summary_name

def main():
    data_folder = "data"
    output_folder = "summaries"
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)
    
    # Get all txt files
    data_path = Path(data_folder)
    doc_files = list(data_path.glob("*.txt"))
    
    print(f"Found {len(doc_files)} documents to summarize")
    print(f"Saving summaries to: {output_folder}/\n")
    
    completed = 0
    failed = 0
    failed_docs = []
    start_time = time.time()
    
    # Process with parallel API calls using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(summarize_document_batch, doc_file): doc_file for doc_file in doc_files}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                success, doc_name, result_data, token_count = result
                
                if success:
                    summary = result_data
                    saved_name = save_summary(doc_name, summary, token_count, output_folder)
                    completed += 1
                    elapsed = time.time() - start_time
                    print(f"[{completed}/{len(doc_files)}] ✓ {doc_name}")
                    print(f"   Tokens: {token_count} (limit: 1500)")
                    print(f"   Saved: {saved_name}")
                    print(f"   Time: {elapsed:.1f}s\n")
                else:
                    failed += 1
                    error = result_data
                    failed_docs.append((doc_name, error))
                    print(f"[{completed + failed}/{len(doc_files)}] ✗ {doc_name}")
                    print(f"   Error: {error}\n")
                
            except Exception as e:
                failed += 1
                print(f"[{completed + failed}/{len(doc_files)}] ✗ Error processing: {e}\n")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"COMPLETE: {completed}/{len(doc_files)} documents summarized")
    print(f"Time: {total_time:.1f}s ({total_time/len(doc_files):.1f}s per document)")
    print(f"Success: {completed}, Failed: {failed}")
    print("="*60)
    
    if failed_docs:
        print("\nFailed documents:")
        for doc_name, error in failed_docs:
            print(f"  - {doc_name}: {error}")

if __name__ == "__main__":
    main()
