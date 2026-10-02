#!/usr/bin/env python3
"""
Simple document structure analysis using keyword matching.
Lists all keyword locations to help navigate the document.
"""

def read_file(filepath):
    """Read the entire document."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text_by_words(text):
    """Split text into words and track their metadata."""
    lines = text.split('\n')
    words = text.split()
    
    chunk_line_numbers = []  # Track which line each word comes from
    
    # Build a mapping of word index to line number
    word_idx = 0
    for line_num, line in enumerate(lines):
        line_words = line.split()
        for word in line_words:
            chunk_line_numbers.append(line_num)
            word_idx += 1
    
    return words, chunk_line_numbers

def create_keyword_masks():
    """Create keyword lists for document orientation."""
    
    keyword_masks = {
        "section_header": [
            "introduction", "overview", "background", "summary", "conclusion",
            "references", "index", "preface", "foreword", "prologue", "epilogue",
            "abstract", "acknowledgments", "appendix", "glossary", "bibliography",
            "contents", "outline", "preamble", "scope", "purpose",
            "objectives", "aims", "goals", "results", "findings", "discussion",
            "recommendations", "next steps", "administration", "management",
        ],
        "chapter_like": [
            "chapter", "book", "part", "section", "subsection", "act", "scene",
            "article", "clause", "provision", "paragraph", "subparagraph",
            "unit", "module", "lesson", "volume", "number", "issue",
        ],
        "formal_title": [
            "title", "heading", "headline", "caption", "label", "name",
            "subject", "topic", "theme", "header", "subtitle",
        ],
        "metadata": [
            "author", "date", "version", "copyright", "published", "editor",
            "publisher", "edition", "page", "pages", "created", "filed",
            "effective", "expiration", "term", "translated", "revised",
            "updated", "modified",
        ],
        "structural_markers": [
            "figure", "table", "list", "exhibit", "schedule", "annex",
            "attachment", "appendix", "supplement", "diagram", "chart",
            "image", "illustration", "note", "footnote", "endnote",
        ],
        "emphasis_markers": [
            "important", "note", "warning", "caution", "attention", "alert",
            "notice", "remember", "key", "critical", "essential", "vital",
            "required", "must", "shall", "should", "mandatory",
        ],
    }
    
    return keyword_masks

def search_with_keywords(text, keyword_masks):
    """Search document for keywords and return all locations."""
    
    words, chunk_line_numbers = chunk_text_by_words(text)
    
    # Convert words to lowercase for matching
    words_lower = [w.lower().strip('.,!?;:\'"()-[]{}') for w in words]
    
    # Search with each mask
    all_matches = []  # Track all matches
    
    for mask_type, keywords in keyword_masks.items():
        # Convert keywords to set for faster lookup
        keyword_set = set(k.lower() for k in keywords)
        
        for i, word in enumerate(words_lower):
            if word in keyword_set:
                original_word = words[i]
                line_num = chunk_line_numbers[i]
                all_matches.append((mask_type, i, line_num, original_word))
    
    return all_matches

def analyze_with_keywords(filepath):
    """Main analysis function using keyword matching."""
    print(f"🔍 Analyzing {filepath} for document structure...\n")
    
    text = read_file(filepath)
    print(f"✓ Document loaded ({len(text)} characters)")
    total_words = len(text.split())
    print(f"✓ Total words: {total_words}\n")
    
    # Create keyword masks
    keyword_masks = create_keyword_masks()
    
    # Search with keywords
    all_matches = search_with_keywords(text, keyword_masks)
    
    # Sort by word position (document order)
    all_matches_sorted = sorted(all_matches, key=lambda x: x[1])
    
    # Print results
    print("="*80)
    print("DOCUMENT STRUCTURE KEYWORDS")
    print("="*80)
    print()
    
    for mask_type, word_idx, line_num, word in all_matches_sorted:
        print(f"Word #{word_idx:6d} | Line {line_num:5d} | [{mask_type:18s}] '{word}'")
    
    # Summary by category
    print("\n" + "="*80)
    print("SUMMARY BY CATEGORY")
    print("="*80)
    
    category_counts = {}
    for mask_type, _, _, _ in all_matches:
        category_counts[mask_type] = category_counts.get(mask_type, 0) + 1
    
    print()
    for mask_type, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{mask_type:18s}: {count:3d} matches")
    
    print(f"\n{'TOTAL':18s}: {len(all_matches):3d} matches")

if __name__ == "__main__":
    filepath = "legal.txt"
    analyze_with_keywords(filepath)
