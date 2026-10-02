import os
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import json

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)


def get_summary(doc_name):
    """Get the summary for a document by reading from file."""
    try:
        summary_file = Path("summaries") / f"{doc_name[:-4]}_summary.txt"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Skip the [TOKENS: X] header if present
                if content.startswith('[TOKENS:'):
                    content = content.split('\n', 1)[1] if '\n' in content else content
                return content.strip()
    except Exception as e:
        pass
    
    return None


def extract_key_terms(query):
    """
    Extract the 3-5 most important search terms that MUST appear in relevant documents.
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": """Extract the 3-5 most important search terms from this query. These are the core concepts that MUST appear in relevant documents. 
Output ONLY the terms, one per line, lowercase, no numbering or extra text."""},
            {"role": "user", "content": f"Query: {query}"}
        ],
        stream=False
    )
    
    key_terms = response.choices[0].message.content.strip().split('\n')
    key_terms = [t.strip().lower() for t in key_terms if t.strip()]
    return key_terms


def semantic_rerank_with_llm(query, document_list):
    """
    Re-rank documents using LLM agent for intelligent ranking.
    Uses LLM for any result set under 100 documents.
    """
    if not document_list or len(document_list) >= 100:
        return document_list
    
    print(f"  Using LLM agent ranking ({len(document_list)} documents)...")
    return rerank_with_llm_agent(query, document_list)


def rerank_with_llm_agent(query, document_list):
    """
    Use LLM agent to rank summaries.
    Feeds all summaries to the agent for intelligent ranking.
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
            print(f"  Warning: No summaries found for documents, using original order")
            return document_list
        
        # Build ranking prompt
        ranking_text = ""
        for i, (doc_name, summary) in enumerate(summaries.items(), 1):
            ranking_text += f"\nDocument {i}: {doc_name}\n"
            ranking_text += f"Summary: {summary[:500]}\n"  # First 500 chars of summary
            ranking_text += "-" * 60 + "\n"
        
        response = client.chat.completions.create(
            model="deepseek-chat",
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
            stream=False
        )
        
        ranked_text = response.choices[0].message.content.strip()
        ranked_names = [line.strip() for line in ranked_text.split('\n') if line.strip()]
        
        # Reorder document_list based on agent ranking
        reranked = []
        for ranked_name in ranked_names:
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
        print(f"  Warning: LLM ranking failed: {e}")
        return document_list



class DocumentCache:
    """Caches document contents to avoid repeated file I/O."""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = Path(data_folder)
        self._cache: Dict[str, str] = {}
        self._load_all_documents()
    
    def _load_all_documents(self):
        """Load all documents into memory on initialization."""
        txt_files = list(self.data_folder.glob("*.txt"))
        print(f"Loading {len(txt_files)} documents into cache...")
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._cache[file_path.name] = f.read()
            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")
        
        print(f"Loaded {len(self._cache)} documents successfully.\n")
    
    def get(self, filename: str) -> Optional[str]:
        """Get document content by filename."""
        return self._cache.get(filename)
    
    def get_all_filenames(self) -> List[str]:
        """Get list of all cached document filenames."""
        return list(self._cache.keys())
    
    def __len__(self):
        return len(self._cache)


class BooleanSearchParser:
    """Robust parser and evaluator for boolean search expressions."""
    
    @staticmethod
    def evaluate(expression: str, content: str) -> bool:
        """Evaluate a boolean search expression against document content."""
        expression = expression.strip()
        content = content.lower()
        
        if not expression:
            return True
        
        # Tokenize the expression
        tokens = BooleanSearchParser._tokenize(expression.lower())
        
        # Parse and evaluate
        try:
            result = BooleanSearchParser._parse_or(tokens, content)
            return result
        except Exception as e:
            print(f"Error evaluating expression '{expression}': {e}")
            return False
    
    @staticmethod
    def _tokenize(expression: str) -> List[str]:
        """Tokenize the boolean expression."""
        # Split on whitespace and parentheses while preserving them
        pattern = r'(\(|\)|\s+)'
        parts = re.split(pattern, expression)
        tokens = [p.strip() for p in parts if p.strip()]
        return tokens
    
    @staticmethod
    def _parse_or(tokens: List[str], content: str) -> bool:
        """Parse OR expressions (lowest precedence)."""
        result = BooleanSearchParser._parse_and(tokens, content)
        
        while tokens and tokens[0] == 'or':
            tokens.pop(0)  # consume 'or'
            right = BooleanSearchParser._parse_and(tokens, content)
            result = result or right
        
        return result
    
    @staticmethod
    def _parse_and(tokens: List[str], content: str) -> bool:
        """Parse AND expressions (higher precedence)."""
        result = BooleanSearchParser._parse_not(tokens, content)
        
        while tokens and tokens[0] == 'and':
            tokens.pop(0)  # consume 'and'
            right = BooleanSearchParser._parse_not(tokens, content)
            result = result and right
        
        return result
    
    @staticmethod
    def _parse_not(tokens: List[str], content: str) -> bool:
        """Parse NOT expressions (highest precedence)."""
        if tokens and tokens[0] == 'not':
            tokens.pop(0)  # consume 'not'
            return not BooleanSearchParser._parse_primary(tokens, content)
        
        return BooleanSearchParser._parse_primary(tokens, content)
    
    @staticmethod
    def _parse_primary(tokens: List[str], content: str) -> bool:
        """Parse primary expressions (terms and parenthesized expressions)."""
        if not tokens:
            return False
        
        token = tokens.pop(0)
        
        # Handle parenthesized expression
        if token == '(':
            result = BooleanSearchParser._parse_or(tokens, content)
            if tokens and tokens[0] == ')':
                tokens.pop(0)  # consume ')'
            return result
        
        # Base case: search term
        return BooleanSearchParser._term_matches(token, content)
    
    @staticmethod
    def _term_matches(term: str, content: str) -> bool:
        """Check if a term matches the content as a whole word."""
        term = term.strip()
        
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
        # Allow punctuation characters around the term: .,-=(){}[]"'
        pattern = r'\b' + re.escape(term) + r'\b'
        return bool(re.search(pattern, content, re.IGNORECASE))


class SearchQueryGenerator:
    """Generates search queries using LLM."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    def generate_initial_query(self, user_query: str, key_terms: List[str] = None) -> str:
        """Generate initial boolean search query from user input, incorporating required key terms."""
        
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
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {user_query}"}
                ],
                temperature=0.3,
                stream=False
            )
            
            expression = response.choices[0].message.content.strip()
            # Clean up any quotes or extra formatting
            expression = expression.replace('"', '').replace("'", '')
            
            # If key terms provided, ensure they're all explicitly in the expression
            if key_terms:
                expression_lower = expression.lower()
                for term in key_terms:
                    # If term not found, add it with AND
                    if term.lower() not in expression_lower:
                        expression = f"({expression}) AND {term}"
            
            return expression
            
        except Exception as e:
            print(f"Error generating query: {e}")
            # Fallback to simple query
            return user_query
    
    def refine_query(self, user_query: str, current_expression: str, 
                     num_results: int, stats: Dict[str, int]) -> str:
        """Refine search query by removing the broadest term to narrow results."""
        
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
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                stream=False
            )
            
            expression = response.choices[0].message.content.strip()
            expression = expression.replace('"', '').replace("'", '')
            
            return expression
            
        except Exception as e:
            print(f"Error refining query: {e}")
            return current_expression


class DocumentSearchEngine:
    """Main search engine that coordinates search operations."""
    
    def __init__(self, api_key: str, data_folder: str = "data"):
        self.cache = DocumentCache(data_folder)
        self.query_gen = SearchQueryGenerator(api_key)
        self.parser = BooleanSearchParser()
    
    def search(self, expression: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Execute a boolean search across all documents.
        Returns matching documents and term statistics.
        """
        matching_docs = []
        
        # Extract terms for statistics
        terms = self._extract_terms(expression)
        term_stats = {term: 0 for term in terms}
        
        # Search all documents
        for filename in self.cache.get_all_filenames():
            content = self.cache.get(filename)
            
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
        """Extract individual search terms from boolean expression."""
        expression = expression.lower()
        # Remove operators and parentheses
        for op in ['(', ')', ' and ', ' or ', ' not ']:
            expression = expression.replace(op, ' ')
        
        # Split and deduplicate
        terms = [t.strip().rstrip('*') for t in expression.split() if t.strip()]
        return list(dict.fromkeys(terms))  # Preserve order while deduping
    
    def smart_search(self, user_query: str, target_count: int = 5, 
                     max_iterations: int = 3, correct_doc: Optional[str] = None) -> List[str]:
        """
        Perform an intelligent agent-driven search with multiple strategies.
        Agent decides which search methods to use (boolean, semantic, summaries).
        Semantic search is used to re-rank boolean results, not replace them.
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
        
        # Step 1: Boolean Search (primary) with key terms
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
        
        # Step 3: Iterative refinement of boolean search if needed
        if len(best_results) > target_count * 2:
            print("STEP 4: Refinement")
            print("-" * 70)
            iteration = 1
            expression_to_refine = expression
            current_docs = matching_docs
            
            while iteration < max_iterations and len(current_docs) > target_count * 2:
                print(f"Refinement iteration {iteration}: Narrowing {len(current_docs)} documents")
                expression_to_refine = self.query_gen.refine_query(
                    user_query, expression_to_refine, len(current_docs), {}
                )
                print(f"New expression: {expression_to_refine}")
                current_docs, _ = self.search(expression_to_refine)
                print(f"✓ Now {len(current_docs)} documents\n")
                
                if correct_doc and correct_doc in current_docs:
                    print(f"✓ Correct document still present\n")
                
                # Re-rank refined results if LLM available
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
        
        if best_results and len(best_results) <= 20:
            print(f"\nResults:\n")
            for i, doc in enumerate(best_results, 1):
                print(f"{i}. {doc}")
        elif best_results:
            print(f"\nShowing first 10 of {len(best_results)} results:\n")
            for i, doc in enumerate(best_results[:10], 1):
                print(f"{i}. {doc}")
            print(f"... and {len(best_results) - 10} more")
        else:
            print("No documents found.")
        
        if correct_doc and best_results:
            if correct_doc in best_results:
                print(f"\n✓ SUCCESS: '{correct_doc}' found!")
            else:
                print(f"\n✗ MISS: '{correct_doc}' not in final results")
        
        print()
        return best_results
    
    def _agent_plan_search(self, user_query: str) -> str:
        """Use LLM to determine optimal search strategy."""
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """You are a search strategy advisor. Given a query, decide which search methods to use:
- Boolean: Good for specific terms, legal concepts, structured queries
- Semantic: Good for conceptual/meaning-based searches, complex requirements
- Iterative refinement: Good when initial results are too broad

Respond with 1-2 sentences explaining your strategy. Be concise."""},
                {"role": "user", "content": f"Query: {user_query}"}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()


def main():
    """Main entry point."""
    # Initialize search engine
    api_key = os.environ["DEEPSEEK_API_KEY"]
    engine = DocumentSearchEngine(api_key)
    
    # Get user input
    user_query = input("Enter your search query: ")
    correct_doc = input("Enter correct document name (or press Enter to skip): ").strip()
    
    # Execute search
    results = engine.smart_search(
        user_query,
        target_count=5,
        max_iterations=3,
        correct_doc=correct_doc if correct_doc else None
    )


if __name__ == "__main__":
    main()