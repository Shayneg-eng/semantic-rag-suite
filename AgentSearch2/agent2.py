import os
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import json
import logging

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# API Configuration
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Search Parameters
DEFAULT_TARGET_COUNT = 5  # Target number of results to return
MAX_REFINEMENT_ITERATIONS = 3  # Maximum iterations for query refinement
TARGET_COUNT_MULTIPLIER = 2  # Trigger refinement when results > target * this
MAX_DOCS_FOR_LLM_RANKING = 100  # Maximum documents to send for LLM ranking

# LLM Parameters
LLM_TEMPERATURE = 0.3  # Temperature for LLM calls
LLM_MAX_TOKENS = 1000  # Max tokens for LLM responses
SUMMARY_TRUNCATION_LENGTH = 500  # Characters of summary to send to LLM

# File Paths
DATA_FOLDER = "data"
SUMMARIES_FOLDER = "summaries"

# Search Result Display
MAX_RESULTS_TO_DISPLAY = 20  # Maximum results to show in full
PREVIEW_RESULTS_COUNT = 10  # Number of results to preview if over max

# Cache Configuration
MAX_CACHE_SIZE_MB = 1000  # Maximum cache size in MB (not enforced, for future use)

# Logging Configuration
LOG_LEVEL = logging.INFO


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# API CLIENT INITIALIZATION
# ============================================================================

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_summary(doc_name: str) -> Optional[str]:
    """
    Get the summary for a document by reading from file.
    
    Args:
        doc_name: Name of the document file
        
    Returns:
        Summary text or None if not found
    """
    try:
        summary_file = Path(SUMMARIES_FOLDER) / f"{doc_name[:-4]}_summary.txt"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Skip the [TOKENS: X] header if present
                if content.startswith('[TOKENS:'):
                    content = content.split('\n', 1)[1] if '\n' in content else content
                return content.strip()
    except Exception as e:
        logger.warning(f"Failed to read summary for {doc_name}: {e}")
    
    return None


def extract_key_terms(query: str) -> List[str]:
    """
    Extract the 3-5 most important search terms that MUST appear in relevant documents.
    
    Args:
        query: User's search query
        
    Returns:
        List of key terms
    """
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Extract the 3-5 most important search terms from this query. These are the core concepts that MUST appear in relevant documents. 
Output ONLY the terms, one per line, lowercase, no numbering or extra text."""
                },
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=LLM_TEMPERATURE,
            stream=False
        )
        
        key_terms = response.choices[0].message.content.strip().split('\n')
        key_terms = [t.strip().lower() for t in key_terms if t.strip()]
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
        
    except Exception as e:
        logger.error(f"Error extracting key terms: {e}")
        # Fallback: simple tokenization
        return [word.lower() for word in query.split() if len(word) > 3][:5]


def semantic_rerank_with_llm(query: str, document_list: List[str]) -> List[str]:
    """
    Re-rank documents using LLM agent for intelligent ranking.
    Uses LLM for any result set under MAX_DOCS_FOR_LLM_RANKING.
    
    Args:
        query: User's search query
        document_list: List of document names to rank
        
    Returns:
        Re-ranked list of document names
    """
    if not document_list or len(document_list) >= MAX_DOCS_FOR_LLM_RANKING:
        return document_list
    
    print(f"  Using LLM agent ranking ({len(document_list)} documents)...")
    return rerank_with_llm_agent(query, document_list)


def rerank_with_llm_agent(query: str, document_list: List[str]) -> List[str]:
    """
    Use LLM agent to rank summaries.
    Feeds all summaries to the agent for intelligent ranking.
    
    Args:
        query: User's search query
        document_list: List of document names to rank
        
    Returns:
        Re-ranked list of document names
    """
    if not document_list:
        return document_list
    
    try:
        # Get summaries for all documents
        summaries = {}
        for doc_name in document_list:
            summary = get_summary(doc_name)
            if summary:
                summaries[doc_name] = summary
        
        if not summaries:
            logger.warning("No summaries found for documents, using original order")
            return document_list
        
        # Build ranking prompt
        ranking_text = ""
        for i, (doc_name, summary) in enumerate(summaries.items(), 1):
            ranking_text += f"\nDocument {i}: {doc_name}\n"
            # Truncate summary to configured length
            truncated_summary = summary[:SUMMARY_TRUNCATION_LENGTH]
            if len(summary) > SUMMARY_TRUNCATION_LENGTH:
                truncated_summary += "..."
            ranking_text += f"Summary: {truncated_summary}\n"
            ranking_text += "-" * 60 + "\n"
        
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a document ranking expert. Given a query and multiple document summaries, 
rank the documents from MOST to LEAST likely to contain the answer to the query.

Output ONLY the document names in ranked order, one per line, starting with the most relevant."""
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nDocuments to rank:\n{ranking_text}\n\nRank these documents from most to least likely to answer the query:"
                }
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=False
        )
        
        ranked_text = response.choices[0].message.content.strip()
        ranked_names = [line.strip() for line in ranked_text.split('\n') if line.strip()]
        
        # Reorder document_list based on agent ranking
        # Use exact matching to avoid false positives
        reranked = []
        for ranked_name in ranked_names:
            # Find exact match in document_list
            for doc_name in document_list:
                if doc_name == ranked_name:
                    if doc_name not in reranked:
                        reranked.append(doc_name)
                    break
            else:
                # If no exact match, try fuzzy matching as fallback
                for doc_name in document_list:
                    if ranked_name in doc_name or doc_name in ranked_name:
                        if doc_name not in reranked:
                            reranked.append(doc_name)
                        break
        
        # Add any documents that weren't ranked (failsafe)
        for doc_name in document_list:
            if doc_name not in reranked:
                reranked.append(doc_name)
        
        print(f"  Top 3 documents by LLM ranking:")
        for i, doc in enumerate(reranked[:3], 1):
            print(f"    {i}. {doc}")
        print()
        
        return reranked
    
    except Exception as e:
        logger.error(f"LLM ranking failed: {e}")
        return document_list


# ============================================================================
# DOCUMENT CACHE CLASS
# ============================================================================

class DocumentCache:
    """Caches document contents to avoid repeated file I/O."""
    
    def __init__(self, data_folder: str = DATA_FOLDER):
        self.data_folder = Path(data_folder)
        self._cache: Dict[str, str] = {}
        self._load_all_documents()
    
    def _load_all_documents(self):
        """Load all documents into memory on initialization."""
        if not self.data_folder.exists():
            logger.error(f"Data folder '{self.data_folder}' does not exist")
            return
        
        txt_files = list(self.data_folder.glob("*.txt"))
        print(f"Loading {len(txt_files)} documents into cache...")
        
        successful_loads = 0
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._cache[file_path.name] = f.read()
                successful_loads += 1
            except Exception as e:
                logger.warning(f"Could not load {file_path.name}: {e}")
        
        print(f"Loaded {successful_loads} documents successfully.\n")
    
    def get(self, filename: str) -> Optional[str]:
        """
        Get document content by filename.
        
        Args:
            filename: Name of the document file
            
        Returns:
            Document content or None if not found
        """
        return self._cache.get(filename)
    
    def get_all_filenames(self) -> List[str]:
        """Get list of all cached document filenames."""
        return list(self._cache.keys())
    
    def __len__(self):
        return len(self._cache)


# ============================================================================
# BOOLEAN SEARCH PARSER CLASS
# ============================================================================

class BooleanSearchParser:
    """Robust parser and evaluator for boolean search expressions."""
    
    @staticmethod
    def evaluate(expression: str, content: str) -> bool:
        """
        Evaluate a boolean search expression against document content.
        
        Args:
            expression: Boolean search expression
            content: Document content to search
            
        Returns:
            True if the expression matches the content
        """
        expression = expression.strip()
        content = content.lower()
        
        if not expression:
            return True
        
        try:
            # Tokenize the expression
            tokens = BooleanSearchParser._tokenize(expression.lower())
            
            # Create a copy to avoid mutation during parsing
            tokens_copy = tokens.copy()
            
            # Parse and evaluate
            result, _ = BooleanSearchParser._parse_or(tokens_copy, content, 0)
            return result
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return False
    
    @staticmethod
    def _tokenize(expression: str) -> List[str]:
        """
        Tokenize the boolean expression.
        
        Args:
            expression: Boolean expression string
            
        Returns:
            List of tokens
        """
        # Split on whitespace and parentheses while preserving them
        pattern = r'(\(|\)|\s+)'
        parts = re.split(pattern, expression)
        tokens = [p.strip() for p in parts if p.strip()]
        return tokens
    
    @staticmethod
    def _parse_or(tokens: List[str], content: str, pos: int) -> Tuple[bool, int]:
        """
        Parse OR expressions (lowest precedence).
        
        Args:
            tokens: List of tokens
            content: Document content
            pos: Current position in token list
            
        Returns:
            Tuple of (result, new_position)
        """
        result, pos = BooleanSearchParser._parse_and(tokens, content, pos)
        
        while pos < len(tokens) and tokens[pos] == 'or':
            pos += 1  # consume 'or'
            right, pos = BooleanSearchParser._parse_and(tokens, content, pos)
            result = result or right
        
        return result, pos
    
    @staticmethod
    def _parse_and(tokens: List[str], content: str, pos: int) -> Tuple[bool, int]:
        """
        Parse AND expressions (higher precedence).
        
        Args:
            tokens: List of tokens
            content: Document content
            pos: Current position in token list
            
        Returns:
            Tuple of (result, new_position)
        """
        result, pos = BooleanSearchParser._parse_not(tokens, content, pos)
        
        while pos < len(tokens) and tokens[pos] == 'and':
            pos += 1  # consume 'and'
            right, pos = BooleanSearchParser._parse_not(tokens, content, pos)
            result = result and right
        
        return result, pos
    
    @staticmethod
    def _parse_not(tokens: List[str], content: str, pos: int) -> Tuple[bool, int]:
        """
        Parse NOT expressions (highest precedence).
        
        Args:
            tokens: List of tokens
            content: Document content
            pos: Current position in token list
            
        Returns:
            Tuple of (result, new_position)
        """
        if pos < len(tokens) and tokens[pos] == 'not':
            pos += 1  # consume 'not'
            result, pos = BooleanSearchParser._parse_primary(tokens, content, pos)
            return not result, pos
        
        return BooleanSearchParser._parse_primary(tokens, content, pos)
    
    @staticmethod
    def _parse_primary(tokens: List[str], content: str, pos: int) -> Tuple[bool, int]:
        """
        Parse primary expressions (terms and parenthesized expressions).
        
        Args:
            tokens: List of tokens
            content: Document content
            pos: Current position in token list
            
        Returns:
            Tuple of (result, new_position)
        """
        if pos >= len(tokens):
            return False, pos
        
        token = tokens[pos]
        
        # Handle parenthesized expression
        if token == '(':
            pos += 1  # consume '('
            result, pos = BooleanSearchParser._parse_or(tokens, content, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1  # consume ')'
            return result, pos
        
        # Base case: search term
        result = BooleanSearchParser._term_matches(token, content)
        return result, pos + 1
    
    @staticmethod
    def _term_matches(term: str, content: str) -> bool:
        """
        Check if a term matches the content as a whole word.
        
        Args:
            term: Search term (may contain wildcards)
            content: Document content (lowercase)
            
        Returns:
            True if term matches
        """
        term = term.strip()
        
        if not term:
            return False
        
        # Handle wildcards with word boundaries
        if '*' in term:
            if term.endswith('*'):
                # Prefix search: "bsp*" matches "bsp", "bsp.", "bsp-", etc. but not "bbsp"
                prefix = re.escape(term[:-1])
                pattern = r'\b' + prefix + r'\w*\b'
                return bool(re.search(pattern, content, re.IGNORECASE))
            else:
                # For other wildcard positions, use regex with word boundaries
                pattern = r'\b' + term.replace('*', r'\w*') + r'\b'
                return bool(re.search(pattern, content, re.IGNORECASE))
        
        # Exact word match with word boundaries (case-insensitive)
        pattern = r'\b' + re.escape(term) + r'\b'
        return bool(re.search(pattern, content, re.IGNORECASE))


# ============================================================================
# SEARCH QUERY GENERATOR CLASS
# ============================================================================

class SearchQueryGenerator:
    """Generates search queries using LLM."""
    
    def __init__(self, api_key: str = DEEPSEEK_API_KEY):
        self.client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL
        )
    
    def generate_initial_query(self, user_query: str, key_terms: List[str] = None) -> str:
        """
        Generate initial boolean search query from user input, incorporating required key terms.
        
        Args:
            user_query: User's natural language query
            key_terms: Optional list of key terms that must appear
            
        Returns:
            Boolean search expression
        """
        system_prompt = """You are a search query expert. Create a boolean search expression that will find relevant documents.

Rules:
- Use OR for synonyms: (term1 OR term2)
- Use AND to connect different concepts
- Use NOT sparingly, only for clearly irrelevant document types
- Keep it simple - prefer broader searches over overly specific ones
- Do NOT use quotes around terms

Output ONLY the boolean search expression, nothing else.

Examples:
User: "supplier nda"
Output: (nda OR confidentiality OR non-disclosure) AND (supplier OR vendor)

User: "employee background check requirements"
Output: (employee OR personnel) AND (background OR verification) AND (check OR screening)"""

        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {user_query}"}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=False
            )
            
            expression = response.choices[0].message.content.strip()
            # Clean up any quotes or extra formatting
            expression = expression.replace('"', '').replace("'", '')
            
            # If key terms provided, ensure they're included in the expression
            if key_terms:
                # Extract existing terms from expression
                existing_terms = set(self._extract_terms_from_expression(expression))
                
                # Add missing key terms
                for term in key_terms:
                    if term.lower() not in existing_terms:
                        expression = f"({expression}) AND {term}"
            
            return expression
            
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            # Fallback to simple query
            return user_query
    
    def _extract_terms_from_expression(self, expression: str) -> List[str]:
        """Extract individual terms from a boolean expression."""
        expression = expression.lower()
        # Remove operators and parentheses
        for op in ['(', ')', ' and ', ' or ', ' not ']:
            expression = expression.replace(op, ' ')
        
        # Split and clean
        terms = [t.strip().rstrip('*') for t in expression.split() if t.strip()]
        return terms
    
    def refine_query(self, user_query: str, current_expression: str, 
                     num_results: int, stats: Dict[str, int]) -> str:
        """
        Refine search query by removing the broadest term to narrow results.
        
        Args:
            user_query: Original user query
            current_expression: Current boolean expression
            num_results: Number of current results
            stats: Term statistics (unused currently)
            
        Returns:
            Refined boolean expression
        """
        system_prompt = """You are a search refinement expert. The current search is returning too many results.

Strategy:
- Remove the BROADEST or MOST GENERAL term from the expression (the one likely matching the most documents)
- Keep all other terms and their relationships intact
- Make ONE targeted change to narrow results
- IMPORTANT: Remove one term, don't replace or restructure the whole expression

Output ONLY the refined boolean search expression, nothing else."""

        user_message = f"""Original query: {user_query}
Current expression: {current_expression}
Results: {num_results} documents (too many - need to narrow)

Remove the broadest/most general term to narrow results:"""

        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=False
            )
            
            expression = response.choices[0].message.content.strip()
            expression = expression.replace('"', '').replace("'", '')
            
            return expression
            
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return current_expression


# ============================================================================
# DOCUMENT SEARCH ENGINE CLASS
# ============================================================================

class DocumentSearchEngine:
    """Main search engine that coordinates search operations."""
    
    def __init__(self, api_key: str = DEEPSEEK_API_KEY, data_folder: str = DATA_FOLDER):
        self.cache = DocumentCache(data_folder)
        self.query_gen = SearchQueryGenerator(api_key)
        self.parser = BooleanSearchParser()
    
    def search(self, expression: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Execute a boolean search across all documents.
        Returns matching documents and term statistics.
        
        Args:
            expression: Boolean search expression
            
        Returns:
            Tuple of (matching documents, term statistics)
        """
        matching_docs = []
        
        # Extract terms for statistics
        terms = self._extract_terms(expression)
        term_stats = {term: 0 for term in terms}
        
        # Search all documents
        for filename in self.cache.get_all_filenames():
            content = self.cache.get(filename)
            
            if content is None:
                continue
            
            # Check if document matches
            if self.parser.evaluate(expression, content):
                matching_docs.append(filename)
            
            # Collect term statistics
            content_lower = content.lower()
            for term in terms:
                if term in content_lower:
                    term_stats[term] += 1
        
        return matching_docs, term_stats
    
    def _extract_terms(self, expression: str) -> List[str]:
        """
        Extract individual search terms from boolean expression.
        
        Args:
            expression: Boolean expression
            
        Returns:
            List of unique terms
        """
        expression = expression.lower()
        # Remove operators and parentheses
        for op in ['(', ')', ' and ', ' or ', ' not ']:
            expression = expression.replace(op, ' ')
        
        # Split and deduplicate
        terms = [t.strip().rstrip('*') for t in expression.split() if t.strip()]
        # Preserve order while deduping
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        return unique_terms
    
    def smart_search(self, user_query: str, target_count: int = DEFAULT_TARGET_COUNT, 
                     max_iterations: int = MAX_REFINEMENT_ITERATIONS, 
                     correct_doc: Optional[str] = None) -> List[str]:
        """
        Perform an intelligent agent-driven search with multiple strategies.
        Agent decides which search methods to use (boolean, semantic, summaries).
        Semantic search is used to re-rank boolean results, not replace them.
        
        Args:
            user_query: User's natural language query
            target_count: Target number of results
            max_iterations: Maximum refinement iterations
            correct_doc: Optional document name for debugging/tracking
            
        Returns:
            List of matching document names
        """
        print(f"{'='*70}")
        print(f"Query: {user_query}")
        if correct_doc:
            print(f"[DEBUG] Tracking: {correct_doc}")
        print(f"{'='*70}\n")
        
        # Extract key terms that MUST appear in results
        print("STEP 0: Extract Key Terms")
        print("-" * 70)
        key_terms = extract_key_terms(user_query)
        print(f"Extracted key terms: {key_terms}")
        print()
        
        # Agent planning step
        print("STEP 1: Agent Planning")
        print("-" * 70)
        plan = self._agent_plan_search(user_query)
        print(f"Plan: {plan}\n")
        
        best_results = []
        search_methods_used = []
        
        # Step 2: Boolean Search (primary) with key terms
        print("STEP 2: Boolean Search (Primary)")
        print("-" * 70)
        expression = self.query_gen.generate_initial_query(user_query, key_terms)
        print(f"LLM-generated expression: {expression}")
        matching_docs, _ = self.search(expression)
        print(f"✓ Found {len(matching_docs)} documents")
        if len(matching_docs) > 0:
            print(f"  First few: {', '.join(matching_docs[:3])}")
        print()
        search_methods_used.append("boolean")
        
        if correct_doc and correct_doc in matching_docs:
            print(f"✓ Correct document found in boolean search\n")
        
        best_results = matching_docs
        
        # Step 3: Use LLM ranking to re-rank boolean results
        if len(matching_docs) > 0:
            print("STEP 3: LLM-based Re-ranking")
            print("-" * 70)
            reranked = semantic_rerank_with_llm(user_query, matching_docs)
            
            search_methods_used.append("LLM re-ranking")
            best_results = reranked
            
            if correct_doc and correct_doc in reranked:
                rank = reranked.index(correct_doc) + 1
                print(f"✓ Correct document ranked #{rank}\n")
        else:
            print("STEP 3: LLM Re-ranking - SKIPPED (no boolean results)\n")
        
        # Step 4: Iterative refinement of boolean search if needed
        if len(best_results) > target_count * TARGET_COUNT_MULTIPLIER:
            print("STEP 4: Refinement")
            print("-" * 70)
            iteration = 1
            expression_to_refine = expression
            current_docs = matching_docs
            
            while iteration <= max_iterations and len(current_docs) > target_count * TARGET_COUNT_MULTIPLIER:
                print(f"Refinement iteration {iteration}: Narrowing {len(current_docs)} documents")
                expression_to_refine = self.query_gen.refine_query(
                    user_query, expression_to_refine, len(current_docs), {}
                )
                print(f"New expression: {expression_to_refine}")
                current_docs, _ = self.search(expression_to_refine)
                print(f"✓ Now {len(current_docs)} documents\n")
                
                if correct_doc and correct_doc in current_docs:
                    print(f"✓ Correct document still present\n")
                elif correct_doc:
                    print(f"⚠ Warning: Correct document lost during refinement\n")
                
                # Re-rank refined results if documents exist
                if len(current_docs) > 0:
                    current_docs = semantic_rerank_with_llm(user_query, current_docs)
                
                iteration += 1
                best_results = current_docs
        else:
            print("STEP 4: Refinement - SKIPPED (already at target range)\n")
        
        # Final results summary
        print("="*70)
        print(f"FINAL RESULTS: {len(best_results)} documents")
        print(f"Search methods used: {', '.join(search_methods_used)}")
        print("="*70)
        
        if best_results and len(best_results) <= MAX_RESULTS_TO_DISPLAY:
            print(f"\nResults:\n")
            for i, doc in enumerate(best_results, 1):
                print(f"{i}. {doc}")
        elif best_results:
            print(f"\nShowing first {PREVIEW_RESULTS_COUNT} of {len(best_results)} results:\n")
            for i, doc in enumerate(best_results[:PREVIEW_RESULTS_COUNT], 1):
                print(f"{i}. {doc}")
            print(f"... and {len(best_results) - PREVIEW_RESULTS_COUNT} more")
        else:
            print("\nNo documents found.")
        
        if correct_doc and best_results:
            if correct_doc in best_results:
                rank = best_results.index(correct_doc) + 1
                print(f"\n✓ SUCCESS: '{correct_doc}' found at rank #{rank}")
            else:
                print(f"\n✗ MISS: '{correct_doc}' not in final results")
        
        print()
        return best_results
    
    def _agent_plan_search(self, user_query: str) -> str:
        """
        Use LLM to determine optimal search strategy.
        
        Args:
            user_query: User's search query
            
        Returns:
            Strategy description
        """
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a search strategy advisor. Given a query, decide which search methods to use:
- Boolean: Good for specific terms, legal concepts, structured queries
- Semantic: Good for conceptual/meaning-based searches, complex requirements
- Iterative refinement: Good when initial results are too broad

Respond with 1-2 sentences explaining your strategy. Be concise."""
                    },
                    {"role": "user", "content": f"Query: {user_query}"}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in agent planning: {e}")
            return "Using default boolean search with LLM re-ranking."


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point."""
    try:
        # Initialize search engine
        engine = DocumentSearchEngine()
        
        if len(engine.cache) == 0:
            print("Error: No documents loaded. Please check the data folder.")
            return
        
        # Get user input
        user_query = input("Enter your search query: ").strip()
        if not user_query:
            print("Error: Empty query provided.")
            return
        
        correct_doc = input("Enter correct document name (or press Enter to skip): ").strip()
        
        # Execute search
        results = engine.smart_search(
            user_query,
            target_count=DEFAULT_TARGET_COUNT,
            max_iterations=MAX_REFINEMENT_ITERATIONS,
            correct_doc=correct_doc if correct_doc else None
        )
        
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()