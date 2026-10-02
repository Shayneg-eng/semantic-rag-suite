"""
Agent Tools: All 14 tools with function calling schemas
"""

import re
import os
import sys
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_manager import get_document_manager
from utils.focus_manager import get_focus_manager
from utils.search_utils import (
    simple_search, boolean_search, fuzzy_search, exact_search,
    semantic_search_query, extract_snippet
)
from settings import PREVIEW_LENGTH


# ============================================================================
# Tool Implementations
# ============================================================================

def tool_search(query: str, mode: str = "simple", scope: str = "all", doc_id: Optional[str] = None) -> Dict[str, Any]:
    """Search for content in documents"""
    doc_manager = get_document_manager()
    focus_manager = get_focus_manager()
    
    # Determine which documents to search
    if doc_id:
        doc_ids = [doc_id] if doc_id in doc_manager.documents else []
    elif scope == "focused":
        doc_ids = focus_manager.get_focused_docs()
    else:
        doc_ids = doc_manager.get_all_doc_ids()
    
    if not doc_ids:
        return {
            "success": False,
            "error": "No documents to search",
            "query": query,
            "mode": mode
        }
    
    # Perform search based on mode
    search_func = {
        "simple": simple_search,
        "boolean": boolean_search,
        "fuzzy": fuzzy_search,
        "exact": exact_search
    }.get(mode, simple_search)
    
    results = []
    for did in doc_ids:
        text = doc_manager.get_document(did)
        if text:
            result = search_func(query, text, did)
            if result['match_count'] > 0:
                results.append(result)
    
    # Auto-fallback to fuzzy if simple returns 0 results
    if mode == "simple" and len(results) == 0:
        print("Simple search returned no results, falling back to fuzzy search...")
        for did in doc_ids:
            text = doc_manager.get_document(did)
            if text:
                result = fuzzy_search(query, text, did)
                if result['match_count'] > 0:
                    results.append(result)
        mode = "fuzzy (auto-fallback)"
    
    # Auto-add to focus for exact mode
    if mode == "exact" and results:
        doc_ids_to_add = [r['doc_id'] for r in results]
        focus_manager.add(doc_ids_to_add, reason=f"Exact match for '{query}'")
    
    # Compact results for display: show first match from each doc
    compact_results = []
    for result in results:
        compact_result = {
            "doc_id": result['doc_id'],
            "match_count": result['match_count'],
            "first_match": result['matches'][0] if result['matches'] else None
        }
        compact_results.append(compact_result)
    
    return {
        "success": True,
        "query": query,
        "mode": mode,
        "total_documents_with_matches": len(results),
        "total_match_count": sum(r['match_count'] for r in results),
        "results": compact_results,
        "note": f"Found {len(results)} documents with {sum(r['match_count'] for r in results)} total matches. Add matching documents to focus to narrow down, or use read_doc to view specific documents."
    }


def tool_semantic_search(query: str, scope: str = "all", top_k: int = 5) -> Dict[str, Any]:
    """Semantic search removed - use keyword search instead"""
    return {
        "success": False,
        "error": "Semantic search is not available. Use 'search' tool instead.",
        "note": "Try: search(query='your terms', mode='simple')"
    }


def tool_list_docs(scope: str = "all") -> Dict[str, Any]:
    """List available documents"""
    doc_manager = get_document_manager()
    focus_manager = get_focus_manager()
    
    if scope == "focused":
        doc_ids = focus_manager.get_focused_docs()
    else:
        doc_ids = doc_manager.get_all_doc_ids()
    
    documents = []
    for doc_id in doc_ids:
        metadata = doc_manager.get_metadata(doc_id)
        if metadata:
            doc_info = {
                "doc_id": doc_id,
                **metadata
            }
            if scope == "focused":
                reason = focus_manager.get_reason(doc_id)
                if reason:
                    doc_info["focus_reason"] = reason
            documents.append(doc_info)
    
    return {
        "success": True,
        "scope": scope,
        "total_documents": len(documents),
        "documents": documents
    }


def tool_add_to_focus(doc_ids: List[str], reason: Optional[str] = None) -> Dict[str, Any]:
    """Add documents to focus list"""
    focus_manager = get_focus_manager()
    doc_manager = get_document_manager()
    
    # Validate document IDs
    valid_ids = []
    invalid_ids = []
    for doc_id in doc_ids:
        if doc_id in doc_manager.documents:
            valid_ids.append(doc_id)
        else:
            invalid_ids.append(doc_id)
    
    added = focus_manager.add(valid_ids, reason)
    
    return {
        "success": True,
        "added": added,
        "already_focused": [d for d in valid_ids if d not in added],
        "invalid": invalid_ids,
        "total_focused": focus_manager.count()
    }


def tool_remove_from_focus(filter_query: Optional[str] = None, doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Remove documents from focus list"""
    focus_manager = get_focus_manager()
    doc_manager = get_document_manager()
    
    if doc_ids:
        # Direct removal
        removed = focus_manager.remove(doc_ids)
        return {
            "success": True,
            "removed": removed,
            "total_focused": focus_manager.count()
        }
    
    elif filter_query:
        # Filter-based removal
        focused_docs = focus_manager.get_focused_docs()
        to_remove = []
        
        query_lower = filter_query.lower()
        
        for doc_id in focused_docs:
            text = doc_manager.get_document(doc_id)
            metadata = doc_manager.get_metadata(doc_id)
            
            if not text or not metadata:
                continue
            
            should_remove = False
            
            # Parse filter patterns
            if "without" in query_lower or "not containing" in query_lower:
                # Extract term to check for absence
                match = re.search(r"without ['\"]?(\w+)['\"]?", query_lower)
                if match:
                    term = match.group(1)
                    if term not in text.lower():
                        should_remove = True
            
            elif "shorter than" in query_lower or "less than" in query_lower:
                # Extract word count threshold
                match = re.search(r"(\d+)\s*words?", query_lower)
                if match:
                    threshold = int(match.group(1))
                    if metadata['word_count'] < threshold:
                        should_remove = True
            
            elif "longer than" in query_lower or "more than" in query_lower:
                # Extract word count threshold
                match = re.search(r"(\d+)\s*words?", query_lower)
                if match:
                    threshold = int(match.group(1))
                    if metadata['word_count'] > threshold:
                        should_remove = True
            
            elif "no dates" in query_lower or "without dates" in query_lower:
                # Check for date patterns
                date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
                if not re.search(date_pattern, text):
                    should_remove = True
            
            if should_remove:
                to_remove.append(doc_id)
        
        removed = focus_manager.remove(to_remove)
        
        return {
            "success": True,
            "filter_query": filter_query,
            "removed": removed,
            "total_focused": focus_manager.count()
        }
    
    else:
        return {
            "success": False,
            "error": "Must provide either filter_query or doc_ids"
        }


def tool_list_focused_docs() -> Dict[str, Any]:
    """Show current focus list"""
    return tool_list_docs(scope="focused")


def tool_clear_focus() -> Dict[str, Any]:
    """Remove all documents from focus"""
    focus_manager = get_focus_manager()
    count = focus_manager.clear()
    
    return {
        "success": True,
        "cleared": count
    }


def tool_read_doc(doc_id: str, mode: str = "preview", chunk_id: Optional[int] = None) -> Dict[str, Any]:
    """Read document content"""
    doc_manager = get_document_manager()
    
    text = doc_manager.get_document(doc_id)
    if not text:
        return {
            "success": False,
            "error": f"Document not found: {doc_id}"
        }
    
    if mode == "preview":
        content = text[:PREVIEW_LENGTH]
        truncated = len(text) > PREVIEW_LENGTH
        if truncated:
            content += "\n\n[... truncated ...]"
        
        return {
            "success": True,
            "doc_id": doc_id,
            "mode": "preview",
            "content": content,
            "truncated": truncated,
            "total_chars": len(text)
        }
    
    elif mode == "full":
        return {
            "success": True,
            "doc_id": doc_id,
            "mode": "full",
            "content": text,
            "total_chars": len(text)
        }
    
    elif mode == "chunk":
        if chunk_id is None:
            return {
                "success": False,
                "error": "chunk_id required for chunk mode"
            }
        
        chunk = doc_manager.get_chunk(doc_id, chunk_id)
        if not chunk:
            return {
                "success": False,
                "error": f"Chunk {chunk_id} not found in {doc_id}"
            }
        
        return {
            "success": True,
            "doc_id": doc_id,
            "mode": "chunk",
            "chunk_id": chunk_id,
            "content": chunk['text']
        }
    
    else:
        return {
            "success": False,
            "error": f"Invalid mode: {mode}"
        }


def tool_get_metadata(doc_ids: Optional[List[str]] = None, scope: str = "all") -> Dict[str, Any]:
    """Get statistics for multiple documents"""
    doc_manager = get_document_manager()
    focus_manager = get_focus_manager()
    
    if doc_ids:
        target_ids = doc_ids
    elif scope == "focused":
        target_ids = focus_manager.get_focused_docs()
    else:
        target_ids = doc_manager.get_all_doc_ids()
    
    metadata_list = []
    for doc_id in target_ids:
        metadata = doc_manager.get_metadata(doc_id)
        if metadata:
            metadata_list.append({
                "doc_id": doc_id,
                **metadata
            })
    
    # Calculate aggregate statistics
    total_words = sum(m['word_count'] for m in metadata_list)
    total_lines = sum(m['line_count'] for m in metadata_list)
    total_chars = sum(m['char_count'] for m in metadata_list)
    
    return {
        "success": True,
        "total_documents": len(metadata_list),
        "aggregate": {
            "total_words": total_words,
            "total_lines": total_lines,
            "total_chars": total_chars,
            "avg_words_per_doc": total_words // len(metadata_list) if metadata_list else 0
        },
        "documents": metadata_list
    }


def tool_get_context(keywords: List[str], scope: str = "all") -> Dict[str, Any]:
    """Find keyword mentions with surrounding context"""
    doc_manager = get_document_manager()
    focus_manager = get_focus_manager()
    
    if scope == "focused":
        doc_ids = focus_manager.get_focused_docs()
    else:
        doc_ids = doc_manager.get_all_doc_ids()
    
    results = {}
    
    for keyword in keywords:
        keyword_results = []
        
        for doc_id in doc_ids:
            text = doc_manager.get_document(doc_id)
            if not text:
                continue
            
            # Find all occurrences
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            matches = list(pattern.finditer(text))
            
            if matches:
                occurrences = []
                for match in matches[:10]:  # Limit to 10 per document
                    occurrences.append({
                        "position": match.start(),
                        "snippet": extract_snippet(text, match.start())
                    })
                
                keyword_results.append({
                    "doc_id": doc_id,
                    "count": len(matches),
                    "occurrences": occurrences
                })
        
        results[keyword] = {
            "total_occurrences": sum(r['count'] for r in keyword_results),
            "documents": keyword_results
        }
    
    return {
        "success": True,
        "keywords": keywords,
        "results": results
    }


def tool_extract_entities(scope: str = "all", entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract structured entities using regex"""
    doc_manager = get_document_manager()
    focus_manager = get_focus_manager()
    
    if entity_types is None:
        entity_types = ["dates", "money", "legal"]
    
    if scope == "focused":
        doc_ids = focus_manager.get_focused_docs()
    else:
        doc_ids = doc_manager.get_all_doc_ids()
    
    # Entity patterns
    patterns = {
        "dates": [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ],
        "money": [
            r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?'
        ],
        "legal": [
            r'\b(?:contract|agreement|liability|plaintiff|defendant|hereby|whereas|jurisdiction|indemnify|confidential)\b'
        ]
    }
    
    results = []
    
    for doc_id in doc_ids:
        text = doc_manager.get_document(doc_id)
        if not text:
            continue
        
        doc_entities = {"doc_id": doc_id}
        
        for entity_type in entity_types:
            if entity_type not in patterns:
                continue
            
            entities = []
            for pattern_str in patterns[entity_type]:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.findall(text)
                entities.extend(matches)
            
            # Remove duplicates and limit
            unique_entities = list(set(entities))[:50]
            doc_entities[entity_type] = {
                "count": len(unique_entities),
                "samples": unique_entities[:20]
            }
        
        results.append(doc_entities)
    
    return {
        "success": True,
        "entity_types": entity_types,
        "total_documents": len(results),
        "results": results
    }


def tool_compare_docs(doc_ids: Optional[List[str]] = None, scope: str = "all") -> Dict[str, Any]:
    """Compare documents for similarities and differences"""
    doc_manager = get_document_manager()
    focus_manager = get_focus_manager()
    
    if doc_ids:
        target_ids = doc_ids
    elif scope == "focused":
        target_ids = focus_manager.get_focused_docs()
    else:
        return {
            "success": False,
            "error": "Must specify doc_ids or use focused scope for comparison"
        }
    
    if len(target_ids) < 2:
        return {
            "success": False,
            "error": "Need at least 2 documents to compare"
        }
    
    if len(target_ids) > 5:
        return {
            "success": False,
            "error": "Maximum 5 documents for comparison"
        }
    
    # Extract word sets
    doc_words = {}
    for doc_id in target_ids:
        text = doc_manager.get_document(doc_id)
        if not text:
            continue
        
        # Extract alphanumeric words, lowercase
        words = set(re.findall(r'\b[a-z0-9]+\b', text.lower()))
        doc_words[doc_id] = words
    
    # Find common words (intersection)
    common_words = set.intersection(*doc_words.values()) if doc_words else set()
    
    # Find unique words per document
    unique_words = {}
    for doc_id, words in doc_words.items():
        other_words = set()
        for other_id, other_word_set in doc_words.items():
            if other_id != doc_id:
                other_words.update(other_word_set)
        
        unique = words - other_words
        unique_words[doc_id] = sorted(list(unique))[:20]  # Top 20
    
    return {
        "success": True,
        "documents": target_ids,
        "common_words": {
            "count": len(common_words),
            "sample": sorted(list(common_words))[:50]
        },
        "unique_words": unique_words
    }


def tool_ask_user(question: str) -> Dict[str, Any]:
    """Request clarification from user"""
    # In real implementation, this would pause and wait for user input
    # For now, we return a placeholder
    return {
        "success": True,
        "question": question,
        "note": "This tool would pause execution and wait for user response"
    }


def tool_submit_answer(answer: str, source_doc_ids: List[str], confidence: str = "medium", excerpts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Submit final answer (terminal action)"""
    return {
        "success": True,
        "final_answer": True,
        "answer": answer,
        "sources": source_doc_ids,
        "confidence": confidence,
        "excerpts": excerpts or []
    }


# ============================================================================
# Tool Registry and Function Calling Schemas
# ============================================================================

TOOL_FUNCTIONS = {
    "search": tool_search,
    "semantic_search": tool_semantic_search,
    "list_docs": tool_list_docs,
    "add_to_focus": tool_add_to_focus,
    "remove_from_focus": tool_remove_from_focus,
    "list_focused_docs": tool_list_focused_docs,
    "clear_focus": tool_clear_focus,
    "read_doc": tool_read_doc,
    "get_metadata": tool_get_metadata,
    "get_context": tool_get_context,
    "extract_entities": tool_extract_entities,
    "compare_docs": tool_compare_docs,
    "ask_user": tool_ask_user,
    "submit_answer": tool_submit_answer
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for content in documents using keyword search. Supports multiple modes: simple (comma-separated keywords), boolean (AND/OR/NOT), fuzzy (typo-tolerant), and exact (phrase matching). Auto-falls back to fuzzy if simple returns no results. Exact mode automatically adds matching documents to focus list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. For simple mode: comma/semicolon-separated terms. For boolean: use AND/OR/NOT operators. For exact: phrase to match."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["simple", "boolean", "fuzzy", "exact"],
                        "description": "Search mode",
                        "default": "simple"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "focused"],
                        "description": "Search in all documents or only focused documents",
                        "default": "all"
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Optional specific document to search in"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_docs",
            "description": "List available documents with metadata (word count, line count, character count)",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["all", "focused"],
                        "description": "List all documents or only focused documents",
                        "default": "all"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_focus",
            "description": "Add documents to the focus list (working set of documents currently interested in). Focus list is persisted across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to add to focus"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional explanation for focusing on these documents"
                    }
                },
                "required": ["doc_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_focus",
            "description": "Remove documents from focus list using either a filter query or direct document IDs. Filter queries support patterns like 'documents without X', 'documents shorter than N words', 'documents with no dates'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_query": {
                        "type": "string",
                        "description": "Natural language filter to select documents to remove (e.g., 'documents without car', 'documents shorter than 1000 words')"
                    },
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific document IDs to remove"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_focused_docs",
            "description": "Show the current focus list with metadata",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_focus",
            "description": "Remove all documents from the focus list",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": "Read document content. Preview mode shows first 2000 characters, full mode shows entire document, chunk mode shows a specific chunk from semantic search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to read"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["preview", "full", "chunk"],
                        "description": "Reading mode",
                        "default": "preview"
                    },
                    "chunk_id": {
                        "type": "integer",
                        "description": "Chunk ID (required if mode is 'chunk')"
                    }
                },
                "required": ["doc_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_metadata",
            "description": "Get statistics (word count, line count, character count) for multiple documents with aggregate totals",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific document IDs to get metadata for"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "focused"],
                        "description": "Get metadata for all documents or only focused documents",
                        "default": "all"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "Find keyword mentions across documents with surrounding context snippets. Useful for seeing how terms are used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keywords to find"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "focused"],
                        "description": "Search in all documents or only focused documents",
                        "default": "all"
                    }
                },
                "required": ["keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract structured entities from documents using regex patterns. Supports dates, money amounts, and legal terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["all", "focused"],
                        "description": "Extract from all documents or only focused documents",
                        "default": "all"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["dates", "money", "legal"]
                        },
                        "description": "Types of entities to extract",
                        "default": ["dates", "money", "legal"]
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_docs",
            "description": "Compare 2-5 documents to find common words and unique words per document. Useful for finding themes and differences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-5 document IDs to compare"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["all", "focused"],
                        "description": "Compare all focused documents if scope is 'focused'",
                        "default": "all"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Request clarification from the user when the query is ambiguous or multiple interpretations exist",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to ask the user"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer to the user. This is a terminal action that ends the agent loop. Always include sources and confidence level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer text"
                    },
                    "source_doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Document IDs used as sources for this answer"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Confidence level in the answer",
                        "default": "medium"
                    },
                    "excerpts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional supporting quotes from documents"
                    }
                },
                "required": ["answer", "source_doc_ids"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given arguments"""
    if tool_name not in TOOL_FUNCTIONS:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Tool execution error: {str(e)}"
        }
