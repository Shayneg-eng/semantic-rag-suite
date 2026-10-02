"""
Search Utilities: Keyword search implementations
"""

import re
import os
import sys
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import (
    MAX_SNIPPET_LENGTH,
    FUZZY_THRESHOLD
)


def extract_snippet(text: str, position: int, max_length: int = MAX_SNIPPET_LENGTH) -> str:
    """Extract a snippet of text around a position"""
    start = max(0, position - max_length // 2)
    end = min(len(text), position + max_length // 2)
    snippet = text[start:end]
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet.strip()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def simple_search(query: str, text: str, doc_id: str) -> Dict[str, Any]:
    """
    Simple keyword search with comma/semicolon-separated terms
    """
    # Split query by comma or semicolon
    terms = [term.strip().lower() for term in re.split(r'[,;]', query)]
    
    matches = []
    text_lower = text.lower()
    
    for term in terms:
        if not term:
            continue
        
        # Find all occurrences
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for match in pattern.finditer(text):
            matches.append({
                'term': term,
                'position': match.start(),
                'snippet': extract_snippet(text, match.start())
            })
    
    return {
        'doc_id': doc_id,
        'match_count': len(matches),
        'matches': matches[:20]  # Limit to first 20 matches
    }


def boolean_search(query: str, text: str, doc_id: str) -> Dict[str, Any]:
    """
    Boolean search with AND/OR/NOT operators
    """
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Simple boolean parser
    # Handle NOT first
    if ' not ' in query_lower:
        parts = query_lower.split(' not ')
        positive_part = parts[0]
        negative_terms = parts[1].split()
        
        # Check if any negative term is present
        for term in negative_terms:
            term = term.strip()
            if term and term in text_lower:
                return {
                    'doc_id': doc_id,
                    'match_count': 0,
                    'matches': []
                }
        
        query_lower = positive_part
    
    # Handle AND
    if ' and ' in query_lower:
        terms = [t.strip() for t in query_lower.split(' and ')]
        
        # All terms must be present
        for term in terms:
            if term and term not in text_lower:
                return {
                    'doc_id': doc_id,
                    'match_count': 0,
                    'matches': []
                }
        
        # Find matches for first term (as representative)
        first_term = terms[0]
        pattern = re.compile(re.escape(first_term), re.IGNORECASE)
        matches = []
        for match in pattern.finditer(text):
            matches.append({
                'term': first_term,
                'position': match.start(),
                'snippet': extract_snippet(text, match.start())
            })
        
        return {
            'doc_id': doc_id,
            'match_count': len(matches),
            'matches': matches[:20]
        }
    
    # Handle OR (or default behavior)
    terms = [t.strip() for t in re.split(r'\s+or\s+', query_lower)]
    matches = []
    
    for term in terms:
        if not term:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for match in pattern.finditer(text):
            matches.append({
                'term': term,
                'position': match.start(),
                'snippet': extract_snippet(text, match.start())
            })
    
    return {
        'doc_id': doc_id,
        'match_count': len(matches),
        'matches': matches[:20]
    }


def fuzzy_search(query: str, text: str, doc_id: str, threshold: float = FUZZY_THRESHOLD) -> Dict[str, Any]:
    """
    Ultra-fast fuzzy search using direct text search with substring matching
    Avoids expensive SequenceMatcher comparisons
    """
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Split into words for analysis
    query_words = [w.strip() for w in query_lower.split() if w.strip() and len(w.strip()) > 2]
    
    if not query_words:
        return {'doc_id': doc_id, 'match_count': 0, 'matches': []}
    
    # Strategy: Look for substrings and near-matches without expensive comparison
    matches = []
    seen_positions = set()
    max_matches = 50  # Limit total matches for performance
    
    for query_word in query_words:
        # First try: exact substring match (fastest)
        pos = 0
        while True:
            pos = text_lower.find(query_word, pos)
            if pos == -1:
                break
            
            if pos not in seen_positions and len(matches) < max_matches:
                seen_positions.add(pos)
                matches.append({
                    'term': query_word,
                    'position': pos,
                    'snippet': extract_snippet(text, pos)
                })
            pos += 1
        
        # Second try: if no exact matches, check for one character off (edit distance 1)
        if not any(m['term'] == query_word for m in matches) and len(matches) < max_matches:
            # Check for variations: missing char, extra char, swapped char
            for i in range(len(query_word) - 1):
                # Try removing each character
                variant = query_word[:i] + query_word[i+1:]
                pos = text_lower.find(variant)
                if pos >= 0 and pos not in seen_positions:
                    seen_positions.add(pos)
                    matches.append({
                        'term': query_word,
                        'matched': variant,
                        'position': pos,
                        'snippet': extract_snippet(text, pos)
                    })
                    break  # Found one, move to next query word
    
    # Return top matches by position order
    matches.sort(key=lambda x: x['position'])
    
    return {
        'doc_id': doc_id,
        'match_count': len(matches),
        'matches': matches[:20]
    }


def exact_search(query: str, text: str, doc_id: str) -> Dict[str, Any]:
    """
    Exact phrase matching
    """
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches = []
    
    for match in pattern.finditer(text):
        matches.append({
            'term': query,
            'position': match.start(),
            'snippet': extract_snippet(text, match.start())
        })
    
    return {
        'doc_id': doc_id,
        'match_count': len(matches),
        'matches': matches[:20]
    }


def semantic_search_query(query: str, chunks: List[Dict[str, Any]], doc_manager=None, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantic search removed - use keyword search instead"""
    return []
