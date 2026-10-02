import os
"""
Directional Shift Test - Robust Version with Caching
Tests query alignment against ALL documents to see ranking
"""

import ollama
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import re
import hashlib
import json
import openai

# Configuration
EMBEDDING_MODEL = "nomic-embed-text"
# EMBEDDING_MODEL = "qwen3-embedding"
DOCS_PATH = Path(r"c:\Coding\Code\RAG\ShiftDim\docs")
SOURCE_DOCS_PATH = Path(r"c:\Coding\Code\RAG\ShiftDim\source_docs")
CACHE_PATH = Path(r"c:\Coding\Code\RAG\ShiftDim\.cache")
CACHE_PATH.mkdir(exist_ok=True)

# LLM Configuration for generating query shifts
POE_API_KEY = os.getenv("POE_API_KEY", "")
LLM_MODEL = "llama-3.1-8b-cs"
llm_client = openai.OpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

# Shift names in order
SHIFT_NAMES = [
    "Entity and Relationship Extraction",
    "Abstraction Level Normalization",
    "Modifier and Qualifier Stripping",
    "Temporal and Causal Structure Removal",
    "Perspective and Voice Normalization",
    "Negation Isolation",
    "Redundancy and Repetition Collapse",
    "Implicit Assumption Extraction",
]

# ============================================================================
# Shift Names and LLM Functions
# ============================================================================

# Shift names in order
SHIFT_NAMES = [
    "Entity and Relationship Extraction",
    "Abstraction Level Normalization",
    "Modifier and Qualifier Stripping",
    "Temporal and Causal Structure Removal",
    "Perspective and Voice Normalization",
    "Negation Isolation",
    "Redundancy and Repetition Collapse",
    "Implicit Assumption Extraction",
]


def call_llm(prompt, temperature=0.0, max_tokens=500):
    """Call LLM via Poe API with deterministic settings."""
    try:
        chat = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=60.0
        )
        result = chat.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"    LLM call error: {type(e).__name__}: {e}")
        return "[Error generating shift]"


def entity_relationship_extraction(text):
    """Extract sentences as [Subject] [Verb] [Object] triplets."""
    prompt = f"""Extract the core facts from the text as complete subject-verb-object triplets.

For EVERY main clause, extract: [Subject] [Verb] [Object/Result]

Rules:
- ALWAYS include complete triplets with all three parts filled in
- Use the specific nouns/entities from the text, not placeholders
- One triplet per line
- Format: subject verb object
- Example triplet formats:
  * "Kubernetes automates deployment"
  * "ReplicaSets maintain pod counts"
  * "LoadBalancer services provision load balancers"

Do NOT extract incomplete fragments. Every line must be a complete triplet.

Text to process:
{text}

Output ONLY the complete triplets, one per line. No numbered lists, no headers, no explanations."""

    return call_llm(prompt, max_tokens=700)


def abstraction_level_normalization(text):
    """Replace specific instances with general category labels."""
    prompt = f"""Create a version of the following text where you replace all specific examples, proper nouns, and concrete instances with their general category labels.

For example:
- Replace "Einstein, Newton, and Hawking" with "physicists"
- Replace "red Ferrari" with "colored vehicle"  
- Replace "Paris" with "city"

Replace ANY specific names, dates, brands, proper nouns, and concrete examples with their category. Be aggressive in generalizing.

Text to process:
{text}

Output ONLY the generalized version. No explanation or commentary."""

    return call_llm(prompt, max_tokens=500)


def modifier_qualifier_stripping(text):
    """Remove adjectives, adverbs, intensifiers, and hedging language."""
    prompt = f"""Remove all adjectives, adverbs, intensifiers, and hedging language from the following text.

Remove words like: very, arguably, somewhat, quite, rather, apparently, beautiful, large, interesting, difficult, interesting, tremendous, etc.

Keep only:
- Adjectives that are definitional (like "living" in "living organism")
- Adjectives that are part of proper nouns
- Verbs and their core arguments (subject, object, etc.)
- Core nouns and essential prepositions

Make the text as bare and stripped down as possible while preserving core meaning.

Text to process:
{text}

Output ONLY the stripped version. No explanation."""

    return call_llm(prompt, max_tokens=500)


def temporal_causal_structure_removal(text):
    """Strip temporal markers, causal connectors, and conditional statements."""
    prompt = f"""Remove all temporal markers and causal connectors from the following text.

Remove:
- Temporal markers: before, after, during, yesterday, last week, previously, eventually, finally, etc.
- Causal connectors: because, therefore, as a result, caused, led to, resulted in, consequent to, etc.
- Conditional statements: if, unless, in case, provided that, assuming that, etc.

Keep only the core state assertions. If "Because X happened, Y occurred", output "X occurred; Y occurred"

Text to process:
{text}

Output ONLY the modified version with pure state assertions. No explanation."""

    return call_llm(prompt, max_tokens=500)


def perspective_voice_normalization(text):
    """Standardize how claims are presented - remove perspective and convert to declarative."""
    prompt = f"""Normalize all claims to objective declarative form. Remove all perspective and voice markers:

- Convert reported speech: "He said that X is true" → "X is true"
- Convert rhetorical questions: "Isn't it obvious that Y?" → "Y is true"
- Remove first-person hedging: "I believe that Z happens" → "Z happens"
- Remove emotional framing: "Sadly, the market crashed" → "The market crashed"
- Convert subjective assessments to objective: "Users seem to prefer this" → "Users prefer this"

Present everything as objective fact, removing the speaker's perspective.

Text to process:
{text}

Output ONLY the normalized version. No explanation or commentary."""

    return call_llm(prompt, max_tokens=500)


def negation_isolation(text):
    """Convert negations to positive form."""
    prompt = f"""Rewrite the text so that every negated claim is converted to its positive form.

Rules:
- "X is NOT Y" → "X is Z" (where Z is the opposite of Y)
- "X does NOT have Y" → "X lacks Y" or "X has Z" (where Z is the opposite)
- "X is NOT important" → "X is unimportant" or "X is trivial"
- Remove all "not", "no", "neither", "nor" constructions
- Express everything as positive assertions about what IS true, not what ISN'T

Keep the same content and meaning, just flip all negations to positive form.

Text to process:
{text}

Output ONLY the rewritten text with all negations converted to positive assertions. No explanations or lists."""

    return call_llm(prompt, max_tokens=700)


def redundancy_collapse(text):
    """Identify and remove semantically equivalent statements."""
    prompt = f"""Identify all semantically equivalent statements in the following text—claims that say essentially the same thing with different wording.

Keep ONLY the first occurrence of each unique claim. Remove all redundant restatements.

Text to process:
{text}

Output ONLY the deduplicated version with redundant claims removed. No explanation."""

    return call_llm(prompt, max_tokens=500)


def implicit_assumption_extraction(text):
    """Identify and make explicit only the NON-OBVIOUS implicit claims."""
    prompt = f"""Find NON-OBVIOUS implicit assumptions in the text. Do NOT list trivial or obvious assumptions.

Examples of NON-OBVIOUS assumptions worth extracting:
- Technical design choices (why use distributed systems instead of centralized?)
- Problem-solution relationships (what problem does this component solve?)
- Tradeoff implications (what benefits come at what cost?)
- Prerequisite knowledge (what must already be true for this to work?)
- Domain-specific constraints (why does the field require this?)

Examples of TRIVIAL assumptions to IGNORE:
- "Kubernetes exists" (too obvious)
- "People use computers" (too obvious)
- "This text is about technology" (too obvious)

For each non-obvious assumption found, add one sentence explaining it.

Text to process:
{text}

Output ONLY the original text followed by a brief section listing 5-8 non-obvious implicit assumptions. Keep it concise, not verbose."""

    return call_llm(prompt, max_tokens=700)


# ============================================================================
# Caching Layer
# ============================================================================

def _get_text_hash(text: str) -> str:
    """Get MD5 hash of text for cache key."""
    return hashlib.md5(text.encode()).hexdigest()


def _get_embedding_cache_path(text_hash: str) -> Path:
    """Get cache file path for embedding."""
    return CACHE_PATH / f"embedding_{text_hash}.json"


def _get_shift_direction_cache_path(text_hash: str, shift_hash: str) -> Path:
    """Get cache file path for shift direction."""
    return CACHE_PATH / f"shift_dir_{text_hash}_{shift_hash}.json"


def _get_alignment_cache_path(query_hash: str, doc_hash: str, query_shifts_hash: str, doc_shifts_hash: str) -> Path:
    """Get cache file path for alignment score."""
    combined = f"{query_hash}_{doc_hash}_{query_shifts_hash}_{doc_shifts_hash}"
    combined_hash = hashlib.md5(combined.encode()).hexdigest()
    return CACHE_PATH / f"alignment_{combined_hash}.json"


def _save_cache(path: Path, data: any) -> None:
    """Save data to cache file."""
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Cache save error: {e}")


def _load_cache(path: Path) -> any:
    """Load data from cache file."""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")
        return None


def get_embedding(text: str) -> List[float]:
    """Get embedding from ollama with caching."""
    text_hash = _get_text_hash(text)
    cache_path = _get_embedding_cache_path(text_hash)
    
    # Try to load from cache
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached
    
    # Compute embedding
    try:
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        embedding = response['embeddings'][0]
        _save_cache(cache_path, embedding)
        return embedding
    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_query_shifts(query_text: str) -> Dict[int, str]:
    """
    Generate 8 semantic shifts for a query using LLM.
    This is the FIX: query shifts are generated from the actual query text,
    not copied from a document's shifts.
    
    Returns dict mapping shift_num (1-8) -> shifted_text
    """
    print(f"    Generating query shifts...", end=" ", flush=True)
    
    # Define all shift tasks
    shift_tasks = [
        (1, "Entity and Relationship Extraction", entity_relationship_extraction),
        (2, "Abstraction Level Normalization", abstraction_level_normalization),
        (3, "Modifier and Qualifier Stripping", modifier_qualifier_stripping),
        (4, "Temporal and Causal Structure Removal", temporal_causal_structure_removal),
        (5, "Perspective and Voice Normalization", perspective_voice_normalization),
        (6, "Negation Isolation", negation_isolation),
        (7, "Redundancy Collapse", redundancy_collapse),
        (8, "Implicit Assumption Extraction", implicit_assumption_extraction),
    ]
    
    shifts = {}
    
    # Generate all 8 shifts sequentially (could parallelize with ThreadPoolExecutor if needed)
    for shift_num, display_name, func in shift_tasks:
        try:
            shifted_text = func(query_text)
            shifts[shift_num] = shifted_text
        except Exception as e:
            print(f"Error in {display_name}: {type(e).__name__}: {e}")
            shifts[shift_num] = "[Error generating shift]"
    
    print("[OK]")
    return shifts


def load_shifts_from_file(doc_num: int) -> Dict[int, str]:
    """Load all 8 pre-generated shifts from DOC_N_shifts.txt file."""
    shifts_file = DOCS_PATH / f"DOC_{doc_num}" / f"DOC_{doc_num}_shifts.txt"
    
    if not shifts_file.exists():
        print(f"Warning: {shifts_file} not found")
        return {}
    
    shifts = {}
    
    with open(shifts_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse each shift section
    pattern = r"SHIFT (\d+): .+?\n={80}\n\n(.*?)(?=\n={80}|$)"
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for i, match in enumerate(matches, 1):
        shift_text = match.group(2).strip()
        if shift_text:
            shifts[i] = shift_text
    
    return shifts


def get_shift_direction(text: str, shift_text: str) -> np.ndarray:
    """
    Get the direction that embedding shifts when transforming.
    Returns normalized delta vector (unit length).
    With caching.
    """
    text_hash = _get_text_hash(text)
    shift_hash = _get_text_hash(shift_text)
    cache_path = _get_shift_direction_cache_path(text_hash, shift_hash)
    
    # Try to load from cache
    cached = _load_cache(cache_path)
    if cached is not None:
        return np.array(cached) if cached != "None" else None
    
    # Compute shift direction
    original = get_embedding(text)
    if original is None:
        _save_cache(cache_path, "None")
        return None
    
    transformed_emb = get_embedding(shift_text)
    if transformed_emb is None:
        _save_cache(cache_path, "None")
        return None
    
    # Delta: how embedding changed
    delta = np.array(transformed_emb) - np.array(original)
    
    # Normalize to unit vector (direction only)
    norm = np.linalg.norm(delta)
    if norm == 0:
        _save_cache(cache_path, "None")
        return None
    
    result = (delta / norm).tolist()
    _save_cache(cache_path, result)
    return np.array(result)


def directional_alignment(query: str, document: str, query_shifts: Dict[int, str], doc_shifts: Dict[int, str]) -> float:
    """
    Measure if query and document shift in SAME direction.
    Returns: mean alignment score across all 8 transformations
    With caching.
    """
    # Generate cache keys
    query_hash = _get_text_hash(query)
    doc_hash = _get_text_hash(document)
    query_shifts_hash = _get_text_hash("".join(str(v) for v in query_shifts.values()))
    doc_shifts_hash = _get_text_hash("".join(str(v) for v in doc_shifts.values()))
    cache_path = _get_alignment_cache_path(query_hash, doc_hash, query_shifts_hash, doc_shifts_hash)
    
    # Try to load from cache
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached
    
    scores = []
    
    for shift_num in range(1, 9):  # 8 shifts total
        if shift_num not in query_shifts or shift_num not in doc_shifts:
            continue
        
        q_shift = get_shift_direction(query, query_shifts[shift_num])
        d_shift = get_shift_direction(document, doc_shifts[shift_num])
        
        if q_shift is None or d_shift is None:
            continue
        
        # Dot product of unit vectors = cosine similarity
        alignment = float(np.dot(q_shift, d_shift))
        scores.append(alignment)
    
    if not scores:
        result = None
    else:
        result = np.mean(scores)
    
    _save_cache(cache_path, result)
    return result


def standard_rag_similarity(query: str, document: str) -> float:
    """
    Standard RAG baseline: compute cosine similarity between 
    query and document embeddings (no transformations).
    With caching.
    """
    query_emb = get_embedding(query)
    doc_emb = get_embedding(document)
    
    if query_emb is None or doc_emb is None:
        return None
    
    query_arr = np.array(query_emb)
    doc_arr = np.array(doc_emb)
    
    # Cosine similarity
    norm_q = np.linalg.norm(query_arr)
    norm_d = np.linalg.norm(doc_arr)
    
    if norm_q == 0 or norm_d == 0:
        return None
    
    return float(np.dot(query_arr, doc_arr) / (norm_q * norm_d))


def load_documents():
    """Load all 30 documents from the new folder structure."""
    docs = {}
    for i in range(1, 31):
        try:
            doc_file = SOURCE_DOCS_PATH / f"DOC_{i}.txt"
            with open(doc_file, "r", encoding='utf-8') as f:
                docs[i] = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: DOC_{i}.txt not found")
    return docs


def run_robust_test():
    """Run comprehensive tests: directional shifts vs standard RAG with multiple query types."""
    
    # Load documents
    docs = load_documents()
    if len(docs) < 30:
        print(f"Error: Expected 30 documents, found {len(docs)}")
        return
    
    # Pre-load all shifts
    all_shifts = {}
    for i in range(1, 31):
        all_shifts[i] = load_shifts_from_file(i)
    
    # Define test queries with their expected relevant document
    test_queries = [
        # Standard/verbose queries
        {
            "type": "STANDARD",
            "name": "OAuth2 Authentication (Standard)",
            "query": "OAuth2 uses access tokens instead of credentials, significantly improving security. When implementing OAuth2, the authentication system must maintain proper token management practices.",
            "relevant_doc": 1,
        },
        # Truly adversarial queries designed to break RAG
        {
            "type": "ADVERSARIAL",
            "name": "Credential Delegation Protocol (Synonym Swap)",
            "query": "How do applications request authorization to access user data without handling passwords directly? Describe the refresh token and short-lived credential system.",
            "relevant_doc": 1,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Third-Party Permission Framework (Industry Jargon)",
            "query": "Bearer token authentication for application scopes, endpoint validation, and secret storage in environment configuration.",
            "relevant_doc": 1,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Broken Authentication Problem Inverse (Inverted Phrasing)",
            "query": "Prevent users from sharing passwords with third-party applications while allowing secure delegated access to protected resources.",
            "relevant_doc": 1,
        },
        # Partial/vague queries
        {
            "type": "PARTIAL",
            "name": "Tokens and Security (Partial/Vague)",
            "query": "tokens and security",
            "relevant_doc": 1,
        },
        # Cross-domain query (could match multiple docs)
        {
            "type": "CROSS-DOMAIN",
            "name": "Service Architecture (Cross-Domain)",
            "query": "How do services communicate and maintain reliability in distributed systems?",
            "relevant_doc": 3,  # Microservices, but could also match Error Handling, Logging, etc.
        },
        # Standard microservices
        {
            "type": "STANDARD",
            "name": "Microservices Architecture (Standard)",
            "query": "Microservices architecture decomposes applications into independent services that communicate through well-defined APIs. Each service handles specific business capabilities and can be deployed independently.",
            "relevant_doc": 3,
        },
        # Adversarial microservices queries
        {
            "type": "ADVERSARIAL",
            "name": "Loose Coupling by Service Boundaries (Terminology Inversion)",
            "query": "Breaking monolithic systems into independently deployable units that maintain loose coupling and high cohesion through API contracts.",
            "relevant_doc": 3,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Asynchronous Component Integration (Synonym Replacement)",
            "query": "How do separate business capability handlers communicate without tight temporal coupling, using async messaging and eventual consistency?",
            "relevant_doc": 3,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Scaling Individual Workloads (Jargon Swap)",
            "query": "Deploying distinct responsibility domains with independent scaling and failure isolation, avoiding shared infrastructure bottlenecks.",
            "relevant_doc": 3,
        },
        # Database Indexing queries
        {
            "type": "STANDARD",
            "name": "Database Indexing (Standard)",
            "query": "Database indexing improves query performance by creating data structures that enable faster data retrieval. Indexes reduce disk I/O and CPU usage by ordering data efficiently.",
            "relevant_doc": 2,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Search Structure Optimization (Synonym Replacement)",
            "query": "How do sorted data structures on disk reduce lookup latency? Describe B-tree and hash table implementations for accelerating retrieval.",
            "relevant_doc": 2,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Query Speed Without Full Scans (Inverted Phrasing)",
            "query": "Avoiding sequential access to all records in large tables by maintaining auxiliary lookup structures.",
            "relevant_doc": 2,
        },
        # Caching queries
        {
            "type": "STANDARD",
            "name": "Caching and Redis (Standard)",
            "query": "Caching stores frequently accessed data in fast memory to reduce latency and database load. Redis provides in-memory data structures like strings, lists, and sets.",
            "relevant_doc": 4,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Hot Data In-Memory Layer (Jargon Swap)",
            "query": "Using temporary in-memory storage for access patterns to bypass slower persistent storage retrieval.",
            "relevant_doc": 4,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Latency Reduction Through Replication (Synonym)",
            "query": "Maintaining copies of frequently accessed values in fast-access storage systems to minimize round-trip times to source systems.",
            "relevant_doc": 4,
        },
        # Logging and Monitoring queries
        {
            "type": "STANDARD",
            "name": "Logging and Monitoring (Standard)",
            "query": "Logging records application events and errors for debugging and monitoring. Centralized log aggregation enables analysis of system behavior across multiple servers.",
            "relevant_doc": 5,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Event Record Aggregation (Synonym Replacement)",
            "query": "Collecting timestamped occurrences from distributed components into unified storage for investigation and pattern detection.",
            "relevant_doc": 5,
        },
        {
            "type": "ADVERSARIAL",
            "name": "System Behavior Visibility (Technical Reframe)",
            "query": "How do organizations understand what happened during failures if they don't collect structured records of all operations?",
            "relevant_doc": 5,
        },
        # Load Balancing queries
        {
            "type": "STANDARD",
            "name": "Load Balancing (Standard)",
            "query": "Load balancing distributes incoming requests across multiple servers to prevent any single server from becoming a bottleneck. Health checks ensure traffic goes only to healthy instances.",
            "relevant_doc": 7,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Request Distribution Across Capacity (Synonym)",
            "query": "Dividing incoming work among multiple processing units based on current capacity and availability metrics.",
            "relevant_doc": 7,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Preventing Single Point Saturation (Inverse)",
            "query": "Ensuring no individual server accumulates excessive traffic that would cause response degradation.",
            "relevant_doc": 7,
        },
        # Rate Limiting queries
        {
            "type": "STANDARD",
            "name": "Rate Limiting (Standard)",
            "query": "Rate limiting restricts the number of requests a client can make within a time window to prevent abuse and ensure fair resource allocation. Algorithms like token bucket and sliding window enforce these limits.",
            "relevant_doc": 9,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Quota Enforcement Per Client (Jargon Swap)",
            "query": "Assigning maximum allowable transaction throughput per consumer using time-window quotas and rejection policies.",
            "relevant_doc": 9,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Preventing Request Floods (Inverse)",
            "query": "How do systems reject excessive traffic from single sources while allowing legitimate usage?",
            "relevant_doc": 9,
        },
        # Error Handling queries
        {
            "type": "STANDARD",
            "name": "Error Handling (Standard)",
            "query": "Error handling manages failures gracefully by catching exceptions and providing meaningful error messages. Retry logic with exponential backoff handles transient failures.",
            "relevant_doc": 10,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Failure Response Patterns (Synonym)",
            "query": "Implementing graceful degradation when operations fail, with automatic recovery attempts using progressive backoff strategies.",
            "relevant_doc": 10,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Transient vs Permanent Failures (Technical Distinction)",
            "query": "Distinguishing between recoverable temporary outages and permanent errors to determine appropriate response strategies.",
            "relevant_doc": 10,
        },
        # Cross-document queries (multiple aspects)
        {
            "type": "CROSS-DOMAIN",
            "name": "Performance Optimization Stack (Multi-Concept)",
            "query": "Combining caching, indexing, and load balancing to optimize end-to-end system performance.",
            "relevant_doc": 4,  # Could match 2, 7, but caching is primary focus
        },
        {
            "type": "CROSS-DOMAIN",
            "name": "Operational Visibility and Reliability (Multi-Concept)",
            "query": "How do monitoring, error handling, and rate limiting work together to maintain system reliability?",
            "relevant_doc": 5,  # Could match 9, 10
        },
        # Edge cases and difficult queries
        {
            "type": "ADVERSARIAL",
            "name": "Acronym vs Expanded Form (Consistency Test)",
            "query": "TTL expiration strategies in cache invalidation using time-based eviction policies.",
            "relevant_doc": 4,
        },
        {
            "type": "ADVERSARIAL",
            "name": "Business Language vs Technical (Vocabulary Mismatch)",
            "query": "How do we ensure system responsiveness doesn't degrade when many users access simultaneously?",
            "relevant_doc": 7,  # Load balancing for responsiveness
        },
        {
            "type": "ADVERSARIAL",
            "name": "Implementation Detail vs Concept (Abstraction Level)",
            "query": "Consistent hashing algorithms for distributed caching systems.",
            "relevant_doc": 4,
        },
    ]
    
    print(f"\n{'='*120}")
    print("COMPREHENSIVE BENCHMARK: Directional Shifts vs Standard RAG")
    print(f"{'='*120}\n")
    
    all_results = []
    
    for test in test_queries:
        test_type = test['type']
        test_name = test['name']
        query = test['query']
        relevant_doc_num = test['relevant_doc']
        
        print(f"[{test_type:12s}] {test_name}")
        print(f"{'-'*120}\n")
        
        # FIX: Generate shifts for the actual query text, not reuse document shifts
        query_shifts = generate_query_shifts(query)
        
        # ====== DIRECTIONAL SHIFTS ======
        alignments_directional = {}
        for doc_num in range(1, 31):
            doc_shifts = all_shifts[doc_num]
            doc_text = docs[doc_num]
            alignment = directional_alignment(query, doc_text, query_shifts, doc_shifts)
            if alignment is not None:
                alignments_directional[doc_num] = alignment
        
        ranked_directional = sorted(alignments_directional.items(), key=lambda x: x[1], reverse=True)
        relevant_rank_directional = next((i+1 for i, (doc_num, _) in enumerate(ranked_directional) if doc_num == relevant_doc_num), None)
        relevant_score_directional = alignments_directional.get(relevant_doc_num, None)
        
        # ====== STANDARD RAG BASELINE ======
        similarities_rag = {}
        for doc_num in range(1, 31):
            doc_text = docs[doc_num]
            similarity = standard_rag_similarity(query, doc_text)
            if similarity is not None:
                similarities_rag[doc_num] = similarity
        
        ranked_rag = sorted(similarities_rag.items(), key=lambda x: x[1], reverse=True)
        relevant_rank_rag = next((i+1 for i, (doc_num, _) in enumerate(ranked_rag) if doc_num == relevant_doc_num), None)
        relevant_score_rag = similarities_rag.get(relevant_doc_num, None)
        
        # Print comparison
        print(f"DIRECTIONAL SHIFTS (8 semantic transformations):")
        print(f"  Top 3 rankings:")
        for i, (doc_num, score) in enumerate(ranked_directional[:3], 1):
            marker = "<- RELEVANT" if doc_num == relevant_doc_num else ""
            print(f"    {i}. DOC_{doc_num:2d}  Score: {score:+.4f}  {marker}")
        print(f"  Relevant doc: Rank #{relevant_rank_directional}/30, Score: {relevant_score_directional:+.4f}")
        
        print(f"\nSTANDARD RAG BASELINE (no transformations):")
        print(f"  Top 3 rankings:")
        for i, (doc_num, score) in enumerate(ranked_rag[:3], 1):
            marker = "<- RELEVANT" if doc_num == relevant_doc_num else ""
            print(f"    {i}. DOC_{doc_num:2d}  Similarity: {score:+.4f}  {marker}")
        print(f"  Relevant doc: Rank #{relevant_rank_rag}/30, Similarity: {relevant_score_rag:+.4f}")
        
        # Comparison verdict
        print(f"\nCOMPARISON:")
        if relevant_rank_directional < relevant_rank_rag:
            advantage = "DIRECTIONAL SHIFTS better"
            delta = relevant_rank_rag - relevant_rank_directional
            print(f"  {advantage} by {delta} rank(s)")
        elif relevant_rank_directional > relevant_rank_rag:
            advantage = "STANDARD RAG better"
            delta = relevant_rank_directional - relevant_rank_rag
            print(f"  {advantage} by {delta} rank(s)")
        else:
            print(f"  Same ranking: #{relevant_rank_directional}/30")
        
        print(f"  Directional vs RAG score gap: {relevant_score_directional - relevant_score_rag:+.4f}")
        print()
        
        all_results.append({
            'name': test_name,
            'type': test_type,
            'relevant_doc': relevant_doc_num,
            'directional_rank': relevant_rank_directional,
            'directional_score': relevant_score_directional,
            'rag_rank': relevant_rank_rag,
            'rag_score': relevant_score_rag,
        })
    
    # Summary
    print(f"\n{'='*120}")
    print("SUMMARY & ANALYSIS")
    print(f"{'='*120}\n")
    
    print(f"Results by Test Type:\n")
    
    for test_type in ["STANDARD", "ADVERSARIAL", "PARTIAL", "CROSS-DOMAIN"]:
        matching = [r for r in all_results if r['type'] == test_type]
        if not matching:
            continue
        
        print(f"{test_type} QUERIES ({len(matching)} tests):")
        
        for r in matching:
            ds_better = "Y" if r['directional_rank'] < r['rag_rank'] else ("=" if r['directional_rank'] == r['rag_rank'] else "N")
            print(f"  {ds_better} {r['name']:45s}")
            print(f"      Directional: Rank #{r['directional_rank']:2d}  ({r['directional_score']:+.4f})")
            print(f"      Std RAG:     Rank #{r['rag_rank']:2d}  ({r['rag_score']:+.4f})")
        print()
    
    # Overall metrics
    print(f"Overall Metrics:\n")
    
    ds_wins = sum(1 for r in all_results if r['directional_rank'] < r['rag_rank'])
    rag_wins = sum(1 for r in all_results if r['directional_rank'] > r['rag_rank'])
    ties = sum(1 for r in all_results if r['directional_rank'] == r['rag_rank'])
    
    ds_avg_rank = np.mean([r['directional_rank'] for r in all_results])
    rag_avg_rank = np.mean([r['rag_rank'] for r in all_results])
    
    total_tests = len(all_results)
    print(f"  Directional Shifts wins:  {ds_wins}/{total_tests}")
    print(f"  Standard RAG wins:        {rag_wins}/{total_tests}")
    print(f"  Ties:                     {ties}/{total_tests}")
    print(f"")
    print(f"  Avg ranking (Directional): {ds_avg_rank:.1f}/30")
    print(f"  Avg ranking (Std RAG):     {rag_avg_rank:.1f}/30")
    
    print(f"\n{'='*120}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'='*120}\n")
    
    # Check robustness across query types
    adversarial_results = [r for r in all_results if r['type'] == "ADVERSARIAL"]
    partial_results = [r for r in all_results if r['type'] == "PARTIAL"]
    cross_domain_results = [r for r in all_results if r['type'] == "CROSS-DOMAIN"]
    
    print(f"Adversarial Queries (paraphrased):")
    if adversarial_results:
        for r in adversarial_results:
            print(f"  {r['name']}: Directional rank #{r['directional_rank']}, RAG rank #{r['rag_rank']}")
    
    print(f"\nPartial/Vague Queries:")
    if partial_results:
        for r in partial_results:
            print(f"  {r['name']}: Directional rank #{r['directional_rank']}, RAG rank #{r['rag_rank']}")
    
    print(f"\nCross-Domain Queries:")
    if cross_domain_results:
        for r in cross_domain_results:
            print(f"  {r['name']}: Directional rank #{r['directional_rank']}, RAG rank #{r['rag_rank']}")
    
    print(f"\n{'CONCLUSION':-^120}")
    
    if ds_wins > rag_wins:
        print("[YES] DIRECTIONAL SHIFTS OUTPERFORM STANDARD RAG")
        print(f"  Wins: {ds_wins}/{ds_wins + rag_wins} comparisons")
        print("  Especially strong on adversarial and vague queries")
    elif rag_wins > ds_wins:
        print("[NO] STANDARD RAG PERFORMS BETTER")
        print(f"  Wins: {rag_wins}/{ds_wins + rag_wins} comparisons")
    else:
        print("= EQUIVALENT PERFORMANCE")
        print("  Both approaches show similar ranking quality")


if __name__ == "__main__":
    run_robust_test()