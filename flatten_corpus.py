import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

# Configuration
CORPUS_DIR = Path("LegalBench-RAG/corpus")
BENCHMARKS_DIR = Path("LegalBench-RAG/benchmarks")
FLATTENED_CORPUS_DIR = Path("LegalBench-RAG/corpus_flat")

def flatten_corpus():
    """Copy all documents from subfolders to a single flat folder."""
    
    # Create flat corpus directory
    FLATTENED_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CORPUS_DIR.exists():
        print(f"Error: Corpus directory not found at {CORPUS_DIR}")
        return False
    
    # Track duplicates
    file_mapping = {}  # old_path -> new_filename
    duplicate_count = defaultdict(int)
    
    print("Flattening corpus structure...")
    
    # Walk through all subfolders and copy files
    for root, dirs, files in os.walk(CORPUS_DIR):
        for file in files:
            source_path = Path(root) / file
            relative_to_corpus = source_path.relative_to(CORPUS_DIR)
            
            # Create new filename preserving original name
            new_filename = str(relative_to_corpus).replace("\\", "_").replace("/", "_")
            target_path = FLATTENED_CORPUS_DIR / new_filename
            
            # Handle duplicates
            if target_path.exists():
                base, ext = os.path.splitext(new_filename)
                duplicate_count[new_filename] += 1
                new_filename = f"{base}_{duplicate_count[new_filename]}{ext}"
                target_path = FLATTENED_CORPUS_DIR / new_filename
            
            # Copy file
            shutil.copy2(source_path, target_path)
            file_mapping[str(relative_to_corpus)] = new_filename
            
            if len(file_mapping) % 100 == 0:
                print(f"  Copied {len(file_mapping)} files...")
    
    print(f"✓ Flattened {len(file_mapping)} files to {FLATTENED_CORPUS_DIR}")
    return file_mapping

def update_benchmarks(file_mapping):
    """Update all benchmark JSON files with new flattened paths."""
    
    if not BENCHMARKS_DIR.exists():
        print(f"Error: Benchmarks directory not found at {BENCHMARKS_DIR}")
        return False
    
    benchmark_files = list(BENCHMARKS_DIR.glob("*.json"))
    print(f"\nUpdating {len(benchmark_files)} benchmark files...")
    
    for benchmark_file in benchmark_files:
        print(f"  Processing {benchmark_file.name}...")
        
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        updated_count = 0
        
        # Update all file paths in tests
        for test in data.get("tests", []):
            for snippet in test.get("snippets", []):
                old_path = snippet.get("file_path", "")
                
                # Normalize path separators for lookup
                normalized_old = old_path.replace("\\", "/")
                
                # Find mapping
                for original_path, new_filename in file_mapping.items():
                    original_normalized = original_path.replace("\\", "/")
                    if normalized_old.endswith(original_normalized) or original_normalized in normalized_old:
                        snippet["file_path"] = new_filename
                        updated_count += 1
                        break
        
        # Write updated benchmark
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"    Updated {updated_count} file paths")
    
    print("✓ Updated all benchmark files")
    return True

def main():
    print("=" * 60)
    print("LEGAL BENCH-RAG CORPUS FLATTENING TOOL")
    print("=" * 60)
    
    # Step 1: Flatten corpus
    file_mapping = flatten_corpus()
    if not file_mapping:
        print("Failed to flatten corpus")
        return False
    
    # Step 2: Update benchmarks
    if not update_benchmarks(file_mapping):
        print("Failed to update benchmarks")
        return False
    
    print("\n" + "=" * 60)
    print("SUCCESS! Corpus has been flattened.")
    print("=" * 60)
    print(f"\nNew structure:")
    print(f"  Documents: {FLATTENED_CORPUS_DIR}/")
    print(f"  Benchmarks: {BENCHMARKS_DIR}/ (updated)")
    print(f"\nTotal files processed: {len(file_mapping)}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
