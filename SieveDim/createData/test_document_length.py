import csv
import os
from pathlib import Path
import ollama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'DATA')
CORPUS_DIR = os.path.join(os.path.dirname(__file__), '..', 'corpus_chunks')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'too_long_documents.txt')
CSV_FILE = os.path.join(DATA_DIR, 'query_document_pairs.csv')

EMBEDDING_MODEL = 'nomic-embed-text'
NUM_WORKERS = 20

def test_document_embedding(pair_data):
    """Try to embed a document. Returns (pair_index, query, doc_filename, success, length, error_msg)."""
    pair_index, query, doc_filename = pair_data
    doc_path = Path(CORPUS_DIR) / doc_filename
    
    if not doc_path.exists():
        return pair_index, query, doc_filename, False, 0, f"File not found"
    
    try:
        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        content_length = len(content)
        
        # Try to embed - let the model tell us if it's too long
        try:
            response = ollama.embed(
                model=EMBEDDING_MODEL,
                input=content,
            )
            return pair_index, query, doc_filename, True, content_length, None
        except Exception as e:
            error_msg = str(e)
            return pair_index, query, doc_filename, False, content_length, error_msg
            
    except Exception as e:
        return pair_index, query, doc_filename, False, 0, f"Error reading file: {str(e)}"

def main():
    print("Testing Query-Document Pairs for Length Issues")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found: {CSV_FILE}")
        return
    
    # Read CSV
    query_pairs = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            query_pairs.append((
                idx + 1,
                row['query'],
                row['correct_document']
            ))
    
    print(f"Found {len(query_pairs)} query-document pairs")
    print(f"Testing with {NUM_WORKERS} parallel workers...\n")
    
    # Test each pair with parallel workers
    too_long_docs = []
    failed_docs = {}
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(test_document_embedding, pair): pair for pair in query_pairs}
        
        with tqdm(total=len(query_pairs), desc="Testing") as pbar:
            for future in as_completed(futures):
                pair_num, query, doc_filename, success, length, error = future.result()
                
                if success:
                    success_count += 1
                else:
                    if doc_filename not in failed_docs:
                        failed_docs[doc_filename] = {
                            'length': length,
                            'error': error,
                            'queries': []
                        }
                    failed_docs[doc_filename]['queries'].append(query)
                    too_long_docs.append((pair_num, query, doc_filename, length, error))
                
                pbar.update(1)
    
    # Write report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("RAG DOCUMENT LENGTH TEST REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Query-Document Pairs: {len(query_pairs)}\n")
        f.write(f"Successfully Embedded: {success_count}\n")
        f.write(f"Failed Due to Model Error: {len(too_long_docs)}\n")
        f.write(f"Unique Problem Documents: {len(failed_docs)}\n\n")
        
        if too_long_docs:
            f.write("PROBLEMATIC QUERY-DOCUMENT PAIRS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Pair #':<8} {'Query':<50} {'Document':<30}\n")
            f.write("-" * 80 + "\n")
            
            for pair_num, query, doc, length, error in too_long_docs:
                query_short = query[:47] + "..." if len(query) > 50 else query
                doc_short = doc[:27] + "..." if len(doc) > 30 else doc
                f.write(f"{pair_num:<8} {query_short:<50} {doc_short:<30}\n")
            
            f.write("\n" * 2)
            f.write("DETAILED PROBLEM DOCUMENTS\n")
            f.write("-" * 80 + "\n")
            
            for doc_name in sorted(failed_docs.keys()):
                info = failed_docs[doc_name]
                f.write(f"\nDocument: {doc_name}\n")
                f.write(f"  Content Length: {info['length']:,} characters\n")
                f.write(f"  Error: {info['error']}\n")
                f.write(f"  Appears in {len(info['queries'])} query pair(s):\n")
                
                for query in info['queries'][:5]:  # Show first 5
                    query_short = query[:70]
                    f.write(f"    - {query_short}...\n" if len(query) > 70 else f"    - {query}\n")
                
                if len(info['queries']) > 5:
                    f.write(f"    ... and {len(info['queries']) - 5} more\n")
        else:
            f.write("\nGREAT NEWS! All documents can be embedded successfully.\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(query_pairs)} pairs OK")
    
    if too_long_docs:
        print(f"\n⚠️  Found {len(too_long_docs)} query-document pairs with issues")
        print(f"⚠️  Found {len(failed_docs)} unique problem document(s)")
        print(f"\nProblematic documents:")
        for doc_name in sorted(failed_docs.keys())[:5]:
            info = failed_docs[doc_name]
            print(f"  - {doc_name} ({info['length']:,} chars, {len(info['queries'])} pairs)")
        if len(failed_docs) > 5:
            print(f"  ... and {len(failed_docs) - 5} more")
    else:
        print("\n✓ No problematic documents found!")
    
    print(f"\nReport saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
