import os
"""
Interactive Document Search Agent with Advanced Tools
Advanced batch operations: metadata, context, entities, comparison, fuzzy search, and user interaction
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import asyncio
from openai import AsyncOpenAI
from difflib import SequenceMatcher

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to False to disable semantic search (ollama)
USE_SEMANTIC_SEARCH = False

try:
    if USE_SEMANTIC_SEARCH:
        import ollama
        OLLAMA_AVAILABLE = True
    else:
        OLLAMA_AVAILABLE = False
except ImportError:
    OLLAMA_AVAILABLE = False
    if USE_SEMANTIC_SEARCH:
        print("⚠️  Ollama not available - semantic search disabled")

# ============================================================================
# SETUP
# ============================================================================

async_client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL_NAME = "deepseek-chat"

DOCS_PATH = "LegalBench-RAG\corpus_flat"
VECTOR_DB_PATH = "vector_db.json"

print("✓ Using DeepSeek API (deepseek-chat model)")
if OLLAMA_AVAILABLE:
    print("✓ Ollama available - semantic search enabled")
else:
    print("✗ Semantic search disabled")

# ============================================================================
# VECTOR DATABASE
# ============================================================================

def load_vector_db() -> Dict:
    """Load pre-computed embeddings from vector database"""
    if OLLAMA_AVAILABLE and Path(VECTOR_DB_PATH).exists():
        with open(VECTOR_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def embed_query(query: str) -> np.ndarray:
    """Embed a query using ollama"""
    if not OLLAMA_AVAILABLE:
        return None
    
    try:
        response = ollama.embed(model='nomic-embed-text', input=query)
        return np.array(response['embeddings'][0])
    except Exception as e:
        print(f"⚠️  Embedding error: {e}")
        return None

def semantic_search(query: str, vector_db: Dict, documents: Dict[str, str], top_k: int = 5) -> Dict[str, float]:
    """
    Semantic search using embeddings.
    Returns: {doc_name: similarity_score, ...}
    """
    if not OLLAMA_AVAILABLE:
        return {}
    
    query_embedding = embed_query(query)
    if query_embedding is None:
        return {}
    
    results = {}
    
    for doc_name, doc_embedding in vector_db.items():
        try:
            doc_emb = np.array(doc_embedding)
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            results[doc_name] = similarity
        except:
            continue
    
    # Sort by similarity
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked[:top_k])

# ============================================================================
# DOCUMENT OPERATIONS
# ============================================================================

def load_all_documents() -> Dict[str, str]:
    """Load all documents"""
    docs_dir = Path(DOCS_PATH)
    documents = {}
    for file_path in sorted(docs_dir.glob('*.txt')):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents[file_path.name] = f.read()
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
    return documents

def search_documents(documents: Dict[str, str], search_term: str) -> Dict[str, List[str]]:
    """
    Keyword search for a word/phrase in all documents.
    Returns: {doc_name: [matching_lines], ...}
    """
    results = {}
    search_lower = search_term.lower()
    
    for doc_name, content in documents.items():
        matches = []
        for i, line in enumerate(content.split('\n')):
            if search_lower in line.lower():
                matches.append(line.strip())
        
        if matches:
            results[doc_name] = matches[:3]
    
    return results

def search_in_document(content: str, search_term: str) -> List[str]:
    """
    Search for a term within a single document content.
    Returns: [matching_lines]
    """
    matches = []
    search_lower = search_term.lower()
    
    for line in content.split('\n'):
        if search_lower in line.lower():
            matches.append(line.strip())
    
    return matches[:5]  # Return up to 5 matches

def fuzzy_search(documents: Dict[str, str], search_term: str, threshold: float = 0.6) -> Dict[str, List[str]]:
    """
    Fuzzy matching for typos/similar terms - optimized for performance.
    Uses quick substring checks first, then fuzzy matching only if needed.
    """
    results = {}
    search_lower = search_term.lower()
    
    for doc_name, content in documents.items():
        matches = []
        content_lower = content.lower()
        
        # Quick check: if exact substring exists, add all matching lines
        if search_lower in content_lower:
            for line in content.split('\n'):
                if search_lower in line.lower():
                    matches.append(line.strip())
        else:
            # Only do fuzzy matching if exact substring not found
            # Check line-by-line to avoid comparing entire large documents
            for line in content.split('\n'):
                if len(line.strip()) > 3:  # Skip very short lines
                    # Calculate similarity ratio
                    ratio = SequenceMatcher(None, search_lower, line.lower()[:100]).ratio()
                    if ratio >= threshold:
                        matches.append(line.strip())
        
        if matches:
            results[doc_name] = matches[:3]
    
    return results

def advanced_search(documents: Dict[str, str], query: str) -> Dict[str, List[str]]:
    """
    Advanced search with AND, OR, NOT operators.
    Examples: "TRO AND emergency", "restraining OR injunction", "NOT criminal"
    """
    results = {}
    
    # Parse operators
    terms_and = []
    terms_or = []
    terms_not = []
    
    # Split by operators (case insensitive)
    parts = re.split(r'\s+(AND|OR|NOT)\s+', query, flags=re.IGNORECASE)
    
    current_op = "AND"  # Default
    for i, part in enumerate(parts):
        if part.upper() in ["AND", "OR", "NOT"]:
            current_op = part.upper()
        elif part.strip():
            if current_op == "AND":
                terms_and.append(part.strip().lower())
            elif current_op == "OR":
                terms_or.append(part.strip().lower())
            elif current_op == "NOT":
                terms_not.append(part.strip().lower())
    
    # Search documents
    for doc_name, content in documents.items():
        content_lower = content.lower()
        
        # AND - all terms must be present
        if terms_and and not all(term in content_lower for term in terms_and):
            continue
        
        # NOT - none of these terms
        if terms_not and any(term in content_lower for term in terms_not):
            continue
        
        # OR - at least one term (if OR terms exist)
        if terms_or and not any(term in content_lower for term in terms_or):
            if not terms_and:  # Only apply OR filter if no AND terms
                continue
        
        # Get matching lines
        matches = []
        for line in content.split('\n'):
            line_lower = line.lower()
            matches_line = False
            
            # Check AND terms
            if all(term in line_lower for term in terms_and):
                matches_line = True
            # Check OR terms
            elif any(term in line_lower for term in terms_or):
                matches_line = True
            
            if matches_line and line.strip():
                matches.append(line.strip())
        
        if matches:
            results[doc_name] = matches[:3]
    
    return results

def summarize_document(content: str, max_length: int = 300) -> str:
    """Get first max_length chars as a quick summary"""
    if len(content) > max_length:
        # Try to end at a sentence boundary
        summary = content[:max_length]
        last_period = summary.rfind('.')
        if last_period > max_length // 2:
            summary = content[:last_period + 1]
        return summary + f"\n... ({len(content) - len(summary)} more characters)"
    return content

def list_all_documents(documents: Dict[str, str]) -> str:
    """Get list of all available documents"""
    doc_list = "\n".join(f"  - {name}" for name in sorted(documents.keys()))
    return f"Available documents ({len(documents)} total):\n{doc_list}"

def get_document_preview(content: str, max_chars: int = 500) -> str:
    """Get a preview of document content"""
    if len(content) > max_chars:
        return content[:max_chars] + f"\n... ({len(content) - max_chars} more characters)"
    return content

# ============================================================================
# BATCH OPERATIONS - NEW TOOLS
# ============================================================================

def get_metadata(documents: Dict[str, str], doc_names: List[str]) -> str:
    """Get metadata (stats) for multiple documents without reading full content"""
    results = "DOCUMENT METADATA:\n"
    for doc_name in doc_names:
        if doc_name in documents:
            content = documents[doc_name]
            word_count = len(content.split())
            line_count = len(content.split('\n'))
            char_count = len(content)
            results += f"\n{doc_name}:\n"
            results += f"  - Words: {word_count}\n"
            results += f"  - Lines: {line_count}\n"
            results += f"  - Characters: {char_count}\n"
    return results

def get_context(documents: Dict[str, str], doc_names: List[str], keyword: str) -> str:
    """Find contextual mentions of keyword across multiple documents"""
    results = f"CONTEXT SEARCH FOR '{keyword}':\n"
    found_any = False
    
    for doc_name in doc_names:
        if doc_name in documents:
            content = documents[doc_name]
            matches = []
            for line in content.split('\n'):
                if keyword.lower() in line.lower():
                    matches.append(line.strip())
            
            if matches:
                found_any = True
                results += f"\n{doc_name}: ({len(matches)} matches)\n"
                for match in matches[:3]:
                    results += f"  - {match[:150]}\n"
    
    if not found_any:
        results += f"No mentions of '{keyword}' found in specified documents"
    
    return results

def extract_entities(documents: Dict[str, str], doc_names: List[str]) -> str:
    """Extract entities like dates, dollar amounts, legal terms from multiple documents"""
    results = "EXTRACTED ENTITIES:\n"
    
    # Simple entity patterns
    date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
    dollar_pattern = r'\$[\d,]+(\.?\d{2})?'
    legal_terms = ['agreement', 'contract', 'liability', 'negligence', 'breach', 'duty', 'tort', 'damages']
    
    for doc_name in doc_names:
        if doc_name in documents:
            content = documents[doc_name]
            dates = re.findall(date_pattern, content)
            amounts = re.findall(dollar_pattern, content)
            found_terms = [term for term in legal_terms if term.lower() in content.lower()]
            
            results += f"\n{doc_name}:\n"
            if dates:
                results += f"  Dates: {', '.join(set(dates[:3]))}\n"
            if amounts:
                results += f"  Dollar Amounts: {', '.join(set(amounts[:3]))}\n"
            if found_terms:
                results += f"  Legal Terms: {', '.join(set(found_terms))}\n"
    
    return results

def compare_documents(documents: Dict[str, str], doc_names: List[str]) -> str:
    """Compare multiple documents to identify differences and commonalities"""
    results = f"COMPARING {len(doc_names)} DOCUMENTS:\n"
    
    if len(doc_names) < 2:
        return "Need at least 2 documents to compare"
    
    # Get first 500 chars of each doc as basis for comparison
    contents = {}
    for doc_name in doc_names:
        if doc_name in documents:
            contents[doc_name] = documents[doc_name][:500]
    
    results += f"\nDocuments to compare: {', '.join(doc_names)}\n"
    
    # Find common words
    doc_words = {}
    for doc_name, content in contents.items():
        words = set(word.lower() for word in re.findall(r'\b\w+\b', content))
        doc_words[doc_name] = words
    
    if doc_words:
        all_docs = list(doc_words.keys())
        common = doc_words[all_docs[0]].copy()
        for words in doc_words.values():
            common = common.intersection(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'and', 'or', 'is', 'in', 'to', 'of', 'that', 'this', 'be'}
        common = common - stop_words
        
        if common:
            results += f"\nCommon topics: {', '.join(sorted(common)[:10])}\n"
    
    # Compare lengths
    results += f"\nDocument Lengths:\n"
    for doc_name in doc_names:
        if doc_name in documents:
            results += f"  {doc_name}: {len(documents[doc_name])} chars\n"
    
    return results

def read_multiple(documents: Dict[str, str], doc_names: List[str], max_chars: int = 400) -> str:
    """Read multiple documents side-by-side for comparison"""
    results = "SIDE-BY-SIDE COMPARISON:\n"
    for doc_name in doc_names:
        if doc_name in documents:
            preview = get_document_preview(documents[doc_name], max_chars)
            results += f"\n{'─'*60}\n{doc_name}:\n{'─'*60}\n{preview}\n"
    return results

def search_in_doc(documents: Dict[str, str], doc_name: str, search_term: str) -> str:
    """Search for a term within a specific document"""
    if doc_name not in documents:
        return f"Document '{doc_name}' not found"
    
    content = documents[doc_name]
    matches = []
    
    for line in content.split('\n'):
        if search_term.lower() in line.lower():
            matches.append(line.strip())
    
    if matches:
        result = f"Search results for '{search_term}' in {doc_name} ({len(matches)} matches):\n"
        for match in matches[:5]:
            result += f"  - {match[:150]}\n"
        return result
    else:
        return f"No matches for '{search_term}' in {doc_name}"

async def extract_answer(query: str, doc_name: str, doc_content: str) -> str:
    """Extract exact quote(s) from document that answer the query"""
    prompt = f"""User Question: "{query}"

Document: {doc_name}
Content:
{doc_content}

Extract the EXACT text from this document that answers the user's question.

CRITICAL RULES:
1. Use EXACT WORDING from the document - do NOT paraphrase, reword, or summarize
2. Copy the text character-by-character as it appears in the source
3. For definitions, extract the COMPLETE definition including all conditions and specifics
4. Skip section headers - extract only the substantive content
5. Preserve all legal terminology exactly as written (e.g., "authorized or obligated by law" NOT "authorized or required")
6. Include all relevant context needed to fully answer the question

Return ONLY the extracted text with no labels, commentary, or modifications."""

    response = await async_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a legal document analyst. Your ONLY job is to copy exact text from documents without any modification, paraphrasing, or summarization. Preserve every word exactly as written in the source."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=2000,
        temperature=0.1
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Clean up formatting labels only, preserve all content
    response_text = response_text.replace('ANSWER:', '').replace('QUOTE:', '').replace('EXTRACTED TEXT:', '')
    response_text = response_text.strip('"').strip()
    
    return response_text

# ============================================================================
# AGENT LOOP
# ============================================================================

async def agent_search_session(query: str, documents: Dict[str, str], vector_db: Dict):
    """
    Main agent loop: LLM searches for the right document.
    Agent can: semantic_search, search, advanced_search, summarize, list_docs, read_doc, or submit_answer
    """
    
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR: {query}")
    print(f"{'='*80}\n")
    
    # Track agent state
    search_history = []
    action_count = 0
    max_actions = 20
    
    # Session log for agent to reference
    session_log = []
    
    # List of documents agent wants to focus on
    docs_of_interest = []
    
    # Initial system message
    system_message = """You are an intelligent document search agent with advanced batch analysis capabilities. Your goal is to find the correct document for the user's query efficiently.

IMPORTANT: When specifying actions, use plain text WITHOUT any markdown formatting (no bold, italics, etc.). Write actions as: search:term NOT **search:**term

AVAILABLE ACTIONS:

SEARCH ACTIONS:
1. search:<term1>,<term2>,<term3> - Simple keyword search. Search multiple terms at once (comma or semicolon separated)""" + (
        "\n2. semantic_search:<query> - Semantic search using embeddings. Best for conceptual searches." if OLLAMA_AVAILABLE else ""
    ) + """
2. advanced_search:<AND/OR/NOT query> - Boolean search. Example: "TRO AND emergency"
3. fuzzy_search:<term1>,<term2>,<term3> - Fuzzy matching handles typos. Search multiple terms at once (comma or semicolon separated)
4. quoted_search:<exact phrase> - QUOTED SEARCH: Find exact phrase and AUTO-ADD all matching docs to focus list. Use when user puts search term in quotes.
5. search_in_doc:<doc_name>:<term> - Search within a specific document

BATCH ANALYSIS (Process multiple documents in ONE action):
6. get_metadata:<doc1>,<doc2>,<doc3> - Get word count, line count, character stats for multiple docs
7. get_context:<doc1>,<doc2>:<keyword> - Find mentions of keyword across multiple documents
8. extract_entities:<doc1>,<doc2>,<doc3> - Extract dates, dollar amounts, legal terms from multiple docs
9. compare:<doc1>,<doc2>,<doc3> - Compare documents to find similarities and differences
10. read_multiple:<doc1>,<doc2> - Side-by-side reading of multiple documents

SINGLE DOCUMENT ACTIONS:
11. summarize:<doc_name> - Get 300-char preview
12. read_doc:<doc_name> - Read document (2000-char preview). Quick way to scan document.
13. read_doc_full:<doc_name> - Read COMPLETE document with NO truncation. Use when looking for deeper content.
14. list_docs - See all available document names

FOCUS MANAGEMENT (Narrow your search):
15. add_to_docs:<doc1>,<doc2>,<doc3> - Add documents to your "docs of interest" list for focused analysis
16. list_docs_of_interest - See which documents are in your focus list
17. clear_docs - Clear the docs of interest list
18. search_in_docs:<term> - Search ONLY within your docs of interest list (much faster/cleaner)
19. read_docs:<doc1>,<doc2> - Read specific documents from your docs of interest list

USER INTERACTION:
20. ask_user:<question> - Ask user for clarification when ambiguous

FINAL ACTION:
21. submit_answer:<doc_name> - Submit your final answer
SPECIAL SEARCH HANDLING:
⚠️ QUOTED SEARCH - Your Most Powerful Narrowing Tool:

WHEN TO USE quoted_search:
- User puts search term in quotes (e.g., "Business Day", "Force Majeure")
- You need to find documents containing an exact phrase or term
- You want to quickly narrow from thousands to tens of documents
- AUTO-ADDS all matching documents to your focus list for further analysis

CHAINING quoted_search (Advanced Narrowing):
- After first quoted_search, you can use it AGAIN on the focused documents
- Example: quoted_search:"Business Day" (gets 58 docs) → then quoted_search:"New York" (narrows to subset)
- ONLY chain quoted_search when you're CERTAIN both terms must appear together
- If unsure whether to narrow further, use search_in_docs instead (safer)

WHEN NOT TO USE:
- User didn't use quotes and the query is conceptual/vague
- You're not sure if the exact phrase appears in documents
- First search attempt - try broader search first to understand the landscape

REMEMBER: quoted_search is aggressive narrowing - only use when confident about exact phrase matching

WORKFLOW PATTERN:
1. If user EXPLICITLY quotes a search term in their question (e.g., "Business Day"), use quoted_search
2. Otherwise use """ + ("semantic_search or " if OLLAMA_AVAILABLE else "") + """search to find promising candidates from full corpus
3. Use add_to_docs to collect additional matches into your focus list
4. Use search_in_docs, get_context, compare on your focused list for cleaner results
5. If you find MULTIPLE documents that match the query criteria:
   a. Use search_in_docs or read_docs to examine ALL candidates
   b. If they're genuinely different valid answers, ask_user for clarification
   c. Only submit if ONE document is clearly the best match
6. Use read_doc_full to verify the exact content before submitting
7. VERIFY: Check that your answer logic makes sense given the query
8. Use submit_answer ONLY when confident in a single document

DECISION PRINCIPLES:

1. CONFIDENCE HEURISTICS:
   - Exact phrase matches from query = HIGH confidence. Treat as likely answer.""" + (
        "\n   - Semantic relevance + exact keyword = MEDIUM-HIGH confidence" if OLLAMA_AVAILABLE else ""
    ) + """
   - Fuzzy matches alone = MEDIUM confidence, verify with read_doc
   - ⚠️ CRITICAL: When user asks about "this agreement" or "the document" without specifying which one:
     * If you find 2+ documents that match the criteria, you MUST ask_user for clarification
     * List the matching documents and ask which one they're referring to
     * NEVER submit an arbitrary answer when multiple valid matches exist
   - If you find multiple valid answers, ask_user which specific agreement/document they meant

2. ACTION EFFICIENCY:
   - After each search, consider: "Should I add these results to my docs of interest?"
   - Use docs_of_interest to keep analysis focused and reduce noise
   - Batch operations on a small focused set are much more useful than on full corpus
   - Trust search results - don't compare/analyze if you have a clear winner

3. SEARCH STRATEGY:""" + (
        "\n   - Try semantic_search FIRST for conceptual or natural language queries. It's your best tool for understanding meaning." if OLLAMA_AVAILABLE else ""
    ) + """
   - Use search:<exact_term> when query contains specific legal terms, document names, or structural keywords""" + (
        "\n   - If semantic_search returns strong results, use them - semantic relevance is a powerful signal" if OLLAMA_AVAILABLE else ""
    ) + """
   - Use fuzzy_search if exact keyword search returns 0 results but query has specific terms
   - Use advanced_search when query needs boolean logic (AND/OR/NOT operators)""" + (
        "\n   - Combine strategies: semantic results + keyword confirmation = high confidence" if OLLAMA_AVAILABLE else
        "\n   - Combine strategies: keyword results + focused analysis = high confidence"
    ) + """

4. COMPLETION CRITERIA:
   - Submit answer ONLY when you have ONE clear best match
   - If query is ambiguous ("this agreement", "the document") and you find multiple matches:
     * Do NOT guess or pick arbitrarily
     * Use ask_user to clarify which specific document they mean
   - Before submitting, verify you can explain WHY this document is the answer
   - Don't over-analyze when evidence is clear, but don't under-analyze when multiple valid answers exist

5. DISAMBIGUATION STRATEGY:
   - Watch for vague references: "this agreement", "the contract", "the document"
   - If you find 2+ documents that equally satisfy the query criteria:
     * Create a clear, numbered list showing ALL matching options
     * Include distinguishing details: document names, company names, and KEY DIFFERENCES (e.g., city pairs)
     * Example format:
       "I found 4 agreements with two-city Business Day definitions:
        1. Berkeley Lights: Boston and San Francisco
        2. Bloom Energy: New York and San Francisco
        3. Gold Resource: Denver and London
        4. Lightbridge: New York and London
        Which one are you asking about?"
     * Ask clearly which specific document they're referring to
   - Only submit without asking if ONE document is clearly superior or the user confirms their choice
   - After user clarifies (e.g., "any of them"), you may choose one and proceed

REMEMBER: Your goal is ACCURACY first, efficiency second. Better to ask one clarifying question than to submit the wrong answer. Use docs_of_interest to keep analysis clean and focused."""
    
    search_suggestion = "semantic_search or advanced_search" if OLLAMA_AVAILABLE else "search, advanced_search, or fuzzy_search"
    initial_message = f"""User Query: "{query}"

Available documents: {len(documents)}

What is your first action? Use {search_suggestion} to find promising documents."""
    
    agent_state = "searching"
    
    while action_count < max_actions and agent_state == "searching":
        action_count += 1
        print(f"\n[Action {action_count}] Thinking...")
        
        # Build formatted session log for agent to see history
        formatted_log = ""
        if session_log:
            formatted_log = "\n" + "="*60 + "\nSESSION HISTORY:\n" + "="*60 + "\n"
            for i, log_entry in enumerate(session_log[-5:], 1):  # Show last 5 actions
                formatted_log += f"\n[Step {i}] {log_entry}\n"
            formatted_log += "="*60 + "\n"
        
        # Build conversation history
        messages = [{"role": "user", "content": initial_message + formatted_log}]
        
        for item in search_history:
            messages.append({"role": "assistant", "content": item['action_request']})
            messages.append({"role": "user", "content": item['result']})
        
        # Get next action from LLM
        response = await async_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                *messages
            ],
            stream=False,
            max_tokens=1000
        )
        
        agent_response = response.choices[0].message.content
        print(f"Agent: {agent_response[:250]}...")
        
        # Clean up markdown formatting that interferes with action parsing
        # Remove bold markers (**text**) but preserve the text
        cleaned_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', agent_response)
        
        action_executed = False
        
        # Check for ask_user action
        ask_match = re.search(r'ask_user:\s*(.+?)(?:\n|$)', agent_response, re.IGNORECASE)
        if ask_match:
            question = ask_match.group(1).strip()
            print(f"\n→ Agent asking user: {question}")
            
            user_answer = input(f"\nAgent's question: {question}\nYour answer: ").strip()
            result_text = f"User answered: {user_answer}"
            session_log.append(f"ACTION: ask_user\nQUESTION: {question}\nANSWER: {user_answer}")
            
            search_history.append({
                'action': 'ask_user',
                'question': question,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for get_metadata action
        meta_match = re.search(r'get_metadata:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if meta_match and not action_executed:
            doc_list = [d.strip() for d in meta_match.group(1).split(',')]
            print(f"\n→ Getting metadata for: {', '.join(doc_list)}")
            
            result_text = get_metadata(documents, doc_list)
            session_log.append(f"ACTION: get_metadata({len(doc_list)} docs)")
            
            search_history.append({
                'action': 'get_metadata',
                'docs': doc_list,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for get_context action
        ctx_match = re.search(r'get_context:\s*(.+?):(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if ctx_match and not action_executed:
            doc_list = [d.strip() for d in ctx_match.group(1).split(',')]
            keyword = ctx_match.group(2).strip()
            print(f"\n→ Getting context for '{keyword}' in: {', '.join(doc_list)}")
            
            result_text = get_context(documents, doc_list, keyword)
            session_log.append(f"ACTION: get_context('{keyword}' in {len(doc_list)} docs)")
            
            search_history.append({
                'action': 'get_context',
                'docs': doc_list,
                'keyword': keyword,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for extract_entities action
        ent_match = re.search(r'extract_entities:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if ent_match and not action_executed:
            doc_list = [d.strip() for d in ent_match.group(1).split(',')]
            print(f"\n→ Extracting entities from: {', '.join(doc_list)}")
            
            result_text = extract_entities(documents, doc_list)
            session_log.append(f"ACTION: extract_entities({len(doc_list)} docs)")
            
            search_history.append({
                'action': 'extract_entities',
                'docs': doc_list,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for compare action
        cmp_match = re.search(r'compare:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if cmp_match and not action_executed:
            doc_list = [d.strip() for d in cmp_match.group(1).split(',')]
            print(f"\n→ Comparing documents: {', '.join(doc_list)}")
            
            result_text = compare_documents(documents, doc_list)
            session_log.append(f"ACTION: compare({len(doc_list)} docs)")
            
            search_history.append({
                'action': 'compare',
                'docs': doc_list,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for read_multiple action
        rm_match = re.search(r'read_multiple:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if rm_match and not action_executed:
            doc_list = [d.strip() for d in rm_match.group(1).split(',')]
            print(f"\n→ Reading multiple documents: {', '.join(doc_list)}")
            
            result_text = read_multiple(documents, doc_list)
            session_log.append(f"ACTION: read_multiple({len(doc_list)} docs)")
            
            search_history.append({
                'action': 'read_multiple',
                'docs': doc_list,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for fuzzy_search action (supports multiple terms separated by comma or semicolon)
        fuzz_match = re.search(r'fuzzy_search:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if fuzz_match and not action_executed:
            fuzz_input = fuzz_match.group(1).strip()
            # Split by comma or semicolon to allow multiple searches
            search_terms = [t.strip() for t in re.split(r'[,;]', fuzz_input) if t.strip()]
            
            print(f"\n→ Fuzzy search for: {search_terms}")
            
            all_results = {}
            for search_term in search_terms:
                results = fuzzy_search(documents, search_term)
                all_results.update(results)
            
            if all_results:
                result_text = f"Fuzzy search results for {search_terms}:\n"
                result_docs = []
                for doc_name, matches in list(all_results.items())[:5]:
                    result_text += f"\n{doc_name}:\n"
                    for match in matches:
                        result_text += f"  - {match[:100]}\n"
                    result_docs.append(doc_name)
                session_log.append(f"ACTION: fuzzy_search({len(search_terms)} terms)\nRESULTS: Found {len(result_docs)} docs")
            else:
                result_text = f"No fuzzy matches for {search_terms}"
                session_log.append(f"ACTION: fuzzy_search({len(search_terms)} terms)\nRESULTS: No matches")
            
            search_history.append({
                'action': 'fuzzy_search',
                'terms': search_terms,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for search_in_doc action
        sid_match = re.search(r'search_in_doc:\s*(.+?):(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if sid_match and not action_executed:
            doc_name = sid_match.group(1).strip()
            search_term = sid_match.group(2).strip()
            print(f"\n→ Searching in {doc_name} for: '{search_term}'")
            
            result_text = search_in_doc(documents, doc_name, search_term)
            session_log.append(f"ACTION: search_in_doc('{doc_name}', '{search_term}')")
            
            search_history.append({
                'action': 'search_in_doc',
                'doc': doc_name,
                'term': search_term,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for semantic_search action
        sem_match = re.search(r'semantic_search:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if sem_match and OLLAMA_AVAILABLE:
            search_query = sem_match.group(1).strip()
            print(f"\n→ Semantic search for: '{search_query}'")
            
            if vector_db:
                results = semantic_search(search_query, vector_db, documents, top_k=5)
                if results:
                    result_text = f"Semantic search results for '{search_query}':\n"
                    result_docs = []
                    for doc_name, score in results.items():
                        line = f"  {doc_name} (relevance: {score:.3f})"
                        result_text += line + "\n"
                        result_docs.append(f"{doc_name} ({score:.3f})")
                    
                    session_log.append(f"ACTION: semantic_search('{search_query}')\nRESULTS: {', '.join(result_docs)}")
                else:
                    result_text = "No semantic search results. Try keyword search."
                    session_log.append(f"ACTION: semantic_search('{search_query}')\nRESULTS: No matches")
            else:
                result_text = "Vector database not loaded. Try keyword search."
                session_log.append(f"ACTION: semantic_search('{search_query}')\nRESULTS: No vector DB")
            
            search_history.append({
                'action': 'semantic_search',
                'query': search_query,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for quoted_search action - auto-adds all matching docs to focus list
        quoted_match = re.search(r'quoted_search:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if quoted_match and not action_executed:
            quoted_term = quoted_match.group(1).strip()
            print(f"\n→ Quoted search for: '{quoted_term}' (auto-adding matches to focus list)")
            
            results = search_documents(documents, quoted_term)
            if results:
                # Auto-add all matching documents to docs_of_interest
                matched_docs = []
                for doc_name in results.keys():
                    if doc_name not in docs_of_interest:
                        docs_of_interest.append(doc_name)
                        matched_docs.append(doc_name)
                    else:
                        matched_docs.append(f"{doc_name} (already in list)")
                
                result_text = f"Quoted search for '{quoted_term}':\n"
                result_text += f"Found {len(results)} matching documents - AUTO-ADDED to focus list:\n"
                for doc_name in matched_docs:
                    result_text += f"  ✓ {doc_name}\n"
                
                session_log.append(f"ACTION: quoted_search('{quoted_term}')\nAUTO-ADDED: {len(results)} docs to focus list")
            else:
                result_text = f"No results for quoted search '{quoted_term}'"
                session_log.append(f"ACTION: quoted_search('{quoted_term}')\nRESULTS: No matches")
            
            search_history.append({
                'action': 'quoted_search',
                'term': quoted_term,
                'docs_added': len(results),
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for advanced_search action
        adv_match = re.search(r'advanced_search:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if adv_match and not action_executed:
            adv_query = adv_match.group(1).strip()
            print(f"\n→ Advanced search: {adv_query}")
            
            results = advanced_search(documents, adv_query)
            if results:
                result_text = f"Advanced search results for '{adv_query}':\n"
                result_docs = []
                for doc_name, matches in list(results.items())[:5]:
                    result_text += f"\n{doc_name}:\n"
                    for match in matches:
                        result_text += f"  - {match[:100]}\n"
                    result_docs.append(doc_name)
                
                session_log.append(f"ACTION: advanced_search('{adv_query}')\nRESULTS: Found {len(result_docs)} docs - {', '.join(result_docs[:3])}")
            else:
                result_text = f"No results for advanced search '{adv_query}'"
                session_log.append(f"ACTION: advanced_search('{adv_query}')\nRESULTS: No matches")
            
            search_history.append({
                'action': 'advanced_search',
                'query': adv_query,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for search action (supports multiple terms separated by comma or semicolon)
        search_match = re.search(r'search:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if search_match and not action_executed:
            search_input = search_match.group(1).strip()
            # Split by comma or semicolon to allow multiple searches
            search_terms = [t.strip() for t in re.split(r'[,;]', search_input) if t.strip()]
            
            print(f"\n→ Keyword search for: {search_terms}")
            
            all_results = {}
            for search_term in search_terms:
                results = search_documents(documents, search_term)
                all_results.update(results)
            
            if all_results:
                result_text = f"Search results for {search_terms}:\n"
                result_docs = []
                for doc_name, matches in list(all_results.items())[:5]:
                    result_text += f"\n{doc_name}:\n"
                    for match in matches:
                        result_text += f"  - {match[:100]}\n"
                    result_docs.append(doc_name)
                
                session_log.append(f"ACTION: search({len(search_terms)} terms)\nRESULTS: Found {len(result_docs)} docs")
            else:
                result_text = f"No documents found containing {search_terms}"
                session_log.append(f"ACTION: search({len(search_terms)} terms)\nRESULTS: No matches")
            
            search_history.append({
                'action': 'search',
                'terms': search_terms,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for list_docs_of_interest action (check this BEFORE list_docs)
        if 'list_docs_of_interest' in cleaned_response.lower() and not action_executed:
            print(f"\n→ Listing docs of interest")
            if docs_of_interest:
                result_text = f"Docs of interest ({len(docs_of_interest)}):\n" + "\n".join(f"  - {d}" for d in docs_of_interest)
            else:
                result_text = "No documents in focus list yet. Use add_to_docs to start building your list."
            session_log.append(f"ACTION: list_docs_of_interest\nFOCUS LIST SIZE: {len(docs_of_interest)}")
            
            search_history.append({
                'action': 'list_docs_of_interest',
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for list_docs action (this will not match if list_docs_of_interest already matched)
        elif 'list_docs' in cleaned_response.lower() and action_executed == False:
            print(f"\n→ Listing all documents")
            result_text = list_all_documents(documents)
            session_log.append(f"ACTION: list_docs")
            
            search_history.append({
                'action': 'list_docs',
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for summarize action
        sum_match = re.search(r'summarize:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if sum_match and not action_executed:
            doc_name = sum_match.group(1).strip()
            print(f"\n→ Summarizing: {doc_name}")
            
            if doc_name in documents:
                summary = summarize_document(documents[doc_name])
                result_text = f"Summary of {doc_name}:\n\n{summary}"
                session_log.append(f"ACTION: summarize('{doc_name}')\nRESULT: Document summary retrieved")
            else:
                result_text = f"Document '{doc_name}' not found."
                session_log.append(f"ACTION: summarize('{doc_name}')\nRESULT: Document not found")
            
            search_history.append({
                'action': 'summarize',
                'doc': doc_name,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for read_doc action
        read_match = re.search(r'read_doc:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if read_match and not action_executed:
            doc_name = read_match.group(1).strip()
            print(f"\n→ Reading document (preview): {doc_name}")
            
            if doc_name in documents:
                preview = get_document_preview(documents[doc_name], 2000)
                result_text = f"Content of {doc_name}:\n\n{preview}"
                session_log.append(f"ACTION: read_doc('{doc_name}')\nRESULT: Preview retrieved")
            else:
                result_text = f"Document '{doc_name}' not found."
                session_log.append(f"ACTION: read_doc('{doc_name}')\nRESULT: Document not found")
            
            search_history.append({
                'action': 'read_doc',
                'doc': doc_name,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for read_doc_full action - read complete document without truncation
        read_full_match = re.search(r'read_doc_full:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if read_full_match and not action_executed:
            doc_name = read_full_match.group(1).strip()
            print(f"\n→ Reading FULL document: {doc_name}")
            
            if doc_name in documents:
                result_text = f"FULL CONTENT of {doc_name}:\n\n{documents[doc_name]}"
                session_log.append(f"ACTION: read_doc_full('{doc_name}')\nRESULT: Full document retrieved")
            else:
                result_text = f"Document '{doc_name}' not found."
                session_log.append(f"ACTION: read_doc_full('{doc_name}')\nRESULT: Document not found")
            
            search_history.append({
                'action': 'read_doc_full',
                'doc': doc_name,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for add_to_docs action
        add_match = re.search(r'add_to_docs:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if add_match and not action_executed:
            doc_list = [d.strip() for d in add_match.group(1).split(',')]
            print(f"\n→ Adding to docs of interest: {', '.join(doc_list[:3])}{'...' if len(doc_list) > 3 else ''}")
            
            added = []
            not_found = []
            already_added = []
            
            for doc in doc_list:
                if doc in documents:
                    if doc not in docs_of_interest:
                        docs_of_interest.append(doc)
                        added.append(doc)
                    else:
                        already_added.append(doc)
                else:
                    not_found.append(doc)
            
            result_parts = []
            if added:
                result_parts.append(f"✓ Added {len(added)} document(s) to focus list")
            if already_added:
                result_parts.append(f"⚠️  {len(already_added)} already in list")
            if not_found:
                result_parts.append(f"✗ {len(not_found)} not found: {', '.join(not_found[:2])}{'...' if len(not_found) > 2 else ''}")
            
            result_text = "\n".join(result_parts) + f"\n\nCurrent focus list: {len(docs_of_interest)} document(s)"
            if docs_of_interest:
                result_text += "\nFirst 5: " + ", ".join(docs_of_interest[:5])
            
            session_log.append(f"ACTION: add_to_docs({len(added)} added, {len(docs_of_interest)} total)")
            
            search_history.append({
                'action': 'add_to_docs',
                'docs': added,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for clear_docs action
        if 'clear_docs' in cleaned_response.lower() and not action_executed:
            print(f"\n→ Clearing docs of interest")
            docs_of_interest.clear()
            result_text = "Docs of interest list cleared."
            session_log.append(f"ACTION: clear_docs\nFOCUS LIST SIZE: 0")
            
            search_history.append({
                'action': 'clear_docs',
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for search_in_docs action
        search_in_match = re.search(r'search_in_docs:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if search_in_match and not action_executed:
            search_term = search_in_match.group(1).strip()
            print(f"\n→ Searching for '{search_term}' in docs of interest ({len(docs_of_interest)} docs)")
            
            if not docs_of_interest:
                result_text = "Error: No documents in focus list. Use add_to_docs first."
            else:
                results = {}
                for doc_name in docs_of_interest:
                    if doc_name in documents:
                        matches = search_in_document(documents[doc_name], search_term)
                        if matches:
                            results[doc_name] = matches
                
                if results:
                    result_text = f"Search results for '{search_term}' in {len(docs_of_interest)} docs:\n"
                    for doc_name, matches in results.items():
                        result_text += f"\n{doc_name}:\n"
                        for match in matches[:2]:
                            result_text += f"  - {match[:150]}\n"
                else:
                    result_text = f"No results for '{search_term}' in focused docs"
            
            session_log.append(f"ACTION: search_in_docs('{search_term}' in {len(docs_of_interest)} docs)")
            
            search_history.append({
                'action': 'search_in_docs',
                'term': search_term,
                'docs_searched': len(docs_of_interest),
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for read_docs action
        read_docs_match = re.search(r'read_docs:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if read_docs_match and not action_executed:
            doc_list = [d.strip() for d in read_docs_match.group(1).split(',')]
            print(f"\n→ Reading from docs of interest: {', '.join(doc_list)}")
            
            result_text = ""
            found_docs = []
            for doc_name in doc_list:
                if doc_name in docs_of_interest and doc_name in documents:
                    found_docs.append(doc_name)
                    preview = get_document_preview(documents[doc_name], 800)
                    result_text += f"\n{'='*60}\n{doc_name}\n{'='*60}\n{preview}\n"
                elif doc_name not in docs_of_interest:
                    result_text += f"\nWarning: '{doc_name}' not in focus list\n"
                else:
                    result_text += f"\nError: '{doc_name}' not found\n"
            
            session_log.append(f"ACTION: read_docs({len(found_docs)} docs from focus list)")
            
            search_history.append({
                'action': 'read_docs',
                'docs': found_docs,
                'action_request': agent_response,
                'result': result_text
            })
            action_executed = True
        
        # Check for submit_answer action
        submit_match = re.search(r'submit_answer:\s*(.+?)(?:\n|$)', cleaned_response, re.IGNORECASE)
        if submit_match and not action_executed:
            doc_name = submit_match.group(1).strip()
            print(f"\n✓ SUBMITTED ANSWER: {doc_name}")
            
            if doc_name in documents:
                # Extract clean answer using LLM
                print("\n→ Extracting answer from document...")
                answer = await extract_answer(query, doc_name, documents[doc_name])
                
                # Verify the answer exists in the document
                if len(answer) < 20 or answer.lower() not in documents[doc_name].lower():
                    print("⚠️ Warning: Extracted answer may not match document content")
                
                # Display clean output
                print(f"\n{'='*80}")
                print(f"DOCUMENT: {doc_name}")
                print(f"{'='*80}")
                print(f"\nANSWER:\n\"{answer}\"")
                print(f"\n{'='*80}")
                
                session_log.append(f"ACTION: submit_answer('{doc_name}')\nRESULT: ✓ ANSWER FOUND")
                agent_state = "completed"
            else:
                result_text = f"Error: Document '{doc_name}' not found."
                session_log.append(f"ACTION: submit_answer('{doc_name}')\nRESULT: Document not found")
                search_history.append({
                    'action': 'submit_answer',
                    'doc': doc_name,
                    'action_request': agent_response,
                    'result': result_text
                })
            action_executed = True
        
        if not action_executed:
            print("\n⚠️  Could not parse action. Asking LLM to clarify...")
            session_log.append(f"ACTION: INVALID - Could not parse action")
            search_history.append({
                'action': 'invalid',
                'action_request': agent_response,
                'result': "Invalid action. Use: semantic_search, search, advanced_search, summarize, list_docs, read_doc, or submit_answer"
            })
    
    # Summary
    print(f"\n{'─'*80}")
    print(f"Search Summary:")
    print(f"  Actions taken: {action_count}")
    print(f"  Status: {agent_state}")
    print(f"  History:")
    for log in session_log:
        print(f"    - {log.split(chr(10))[0]}")
    
    return agent_state == "completed"

# ============================================================================
# MAIN INTERACTIVE LOOP
# ============================================================================

async def main():
    print("\n" + "="*80)
    print("INTELLIGENT DOCUMENT SEARCH AGENT")
    print("="*80)
    
    # Load documents
    print("\nLoading documents...")
    documents = load_all_documents()
    print(f"✓ Loaded {len(documents)} documents")
    
    # Load vector database for semantic search
    vector_db = {}
    if OLLAMA_AVAILABLE:
        print("Loading vector database...")
        vector_db = load_vector_db()
        if vector_db:
            print(f"✓ Loaded embeddings for {len(vector_db)} documents")
        else:
            print("⚠️  Vector database not found or empty")
    
    print()
    
    # Interactive loop
    while True:
        query = input("📝 Enter your query (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            print("✓ Goodbye!")
            break
        
        if not query:
            print("Please enter a query.")
            continue
        
        # Run agent search session
        success = await agent_search_session(query, documents, vector_db)
        
        if not success:
            print("\n⚠️  Agent could not find a definitive answer.")

if __name__ == "__main__":
    asyncio.run(main())
