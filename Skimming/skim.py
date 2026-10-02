#!/usr/bin/env python3
"""
Intelligent document skimming using DeepSeek API.
Allows an LLM to navigate through large documents to find information efficiently.
Supports AND/OR operators in search queries.
"""

import os
import re
from openai import OpenAI

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY', os.environ["DEEPSEEK_API_KEY"]),
    base_url="https://api.deepseek.com"
)

def read_file(filepath):
    """Read the entire document."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

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

def search_keywords(text):
    """Search document for structural keywords and return locations."""
    
    words = text.split()
    lines = text.split('\n')
    
    # Build word to line mapping
    word_idx = 0
    word_to_line = {}
    for line_num, line in enumerate(lines):
        line_words = line.split()
        for _ in line_words:
            word_to_line[word_idx] = line_num
            word_idx += 1
    
    # Convert words to lowercase for matching
    words_lower = [w.lower().strip('.,!?;:\'"()-[]{}') for w in words]
    
    # Search for keywords
    keyword_masks = create_keyword_masks()
    keyword_locations = {}  # mask_type -> list of word indices
    
    for mask_type, keywords in keyword_masks.items():
        keyword_locations[mask_type] = []
        keyword_set = set(k.lower() for k in keywords)
        
        for i, word in enumerate(words_lower):
            if word in keyword_set:
                keyword_locations[mask_type].append((i, word_to_line.get(i, 0)))
    
    return keyword_locations

def format_keyword_context(keyword_locations):
    """Format keyword locations for LLM context."""
    
    context = "=== DOCUMENT STRUCTURE KEYWORDS ===\n\n"
    
    # Group by mask type and show locations
    for mask_type, locations in sorted(keyword_locations.items()):
        if locations:
            word_positions = [str(pos) for pos, _ in locations]
            context += f"Word # {', '.join(word_positions)} [{mask_type:18s}]\n"
    
    context += "\n"
    return context

def parse_search_query(search_query):
    """
    Parse search query for AND/OR operators.
    Returns a dict with parsed structure.
    
    Examples:
    - "find X AND Y" -> {"type": "AND", "terms": ["X", "Y"]}
    - "find X OR Y OR Z" -> {"type": "OR", "terms": ["X", "Y", "Z"]}
    - "find X" -> {"type": "SIMPLE", "terms": ["X"]}
    """
    
    # Normalize query
    query = search_query.strip()
    
    # Check for AND operator
    if ' AND ' in query.upper():
        parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        terms = [p.strip() for p in parts]
        return {"type": "AND", "terms": terms}
    
    # Check for OR operator
    elif ' OR ' in query.upper():
        parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
        terms = [p.strip() for p in parts]
        return {"type": "OR", "terms": terms}
    
    # Simple query
    else:
        return {"type": "SIMPLE", "terms": [query]}

def format_parsed_query(parsed_query):
    """Format parsed query for display."""
    if parsed_query["type"] == "SIMPLE":
        return f"Search: {parsed_query['terms'][0]}"
    else:
        operator = parsed_query["type"]
        terms_str = f" {operator} ".join(parsed_query['terms'])
        return f"Search ({operator}): {terms_str}"

def get_initial_context(text, num_initial_lines=50, num_samples=25):
    """
    Get initial context: first N lines + evenly dispersed samples throughout.
    Returns formatted string with character positions labeled.
    """
    lines = text.split('\n')
    initial_lines = lines[:num_initial_lines]
    
    # Get evenly dispersed samples from the rest
    remaining_lines = lines[num_initial_lines:]
    step = max(1, len(remaining_lines) // num_samples)
    sample_lines = remaining_lines[::step][:num_samples]
    
    # Build context string with character positions
    context = "=== INITIAL DOCUMENT OVERVIEW ===\n"
    context += f"Total document length: {len(text)} characters\n"
    context += f"Total lines: {len(lines)}\n\n"
    
    context += "--- FIRST 50 LINES (chars 0-) ---\n"
    char_pos = 0
    for i, line in enumerate(initial_lines):
        context += f"[Line {i+1}, Char {char_pos}] {line}\n"
        char_pos += len(line) + 1  # +1 for newline
    
    context += "\n--- EVENLY DISPERSED SAMPLES ---\n"
    char_pos = len('\n'.join(initial_lines)) + 1
    for i, line in enumerate(sample_lines):
        line_idx = num_initial_lines + (i * step)
        char_pos = sum(len(l) + 1 for l in lines[:line_idx])
        context += f"[Line {line_idx+1}, Char {char_pos}] {line}\n"
    
    return context

def get_text_range(text, char_pos, range_size=1000):
    """
    Get text around a specific character position.
    Returns text from (char_pos - range_size) to (char_pos + range_size).
    """
    start = max(0, char_pos - range_size)
    end = min(len(text), char_pos + range_size)
    return text[start:end]

def skim_document(filepath, search_query, max_iterations=10):
    """
    Main skimming function that uses DeepSeek to navigate the document.
    Supports AND/OR operators in search queries.
    
    Usage:
    - Simple: skim_document(path, "What is X?")
    - AND: skim_document(path, "What is Pierre AND where does he live?")
    - OR: skim_document(path, "Who are the main characters OR what is the setting?")
    """
    print(f"\n🔍 Starting document skim for: '{search_query}'")
    
    # Parse search query
    parsed_query = parse_search_query(search_query)
    print(f"📋 Query type: {format_parsed_query(parsed_query)}")
    
    print(f"📄 Reading: {filepath}")
    
    # Read document
    document_text = read_file(filepath)
    print(f"✓ Document loaded ({len(document_text)} characters)")
    
    # Search for structural keywords
    print("📊 Scanning for document structure keywords...")
    keyword_locations = search_keywords(document_text)
    keyword_context = format_keyword_context(keyword_locations)
    print("✓ Document structure identified\n")
    
    # Get initial context
    initial_context = get_initial_context(document_text)
    
    # Build system prompt based on query type
    if parsed_query["type"] == "AND":
        search_instruction = f"""You are searching for content that contains ALL of these topics:
{chr(10).join('- ' + t for t in parsed_query['terms'])}

Find a section that discusses all of these topics together, or explains their relationship."""
    elif parsed_query["type"] == "OR":
        search_instruction = f"""You are searching for content that contains ANY of these topics:
{chr(10).join('- ' + t for t in parsed_query['terms'])}

Find a section that discusses at least one of these topics."""
    else:
        search_instruction = f"Find information about: {parsed_query['terms'][0]}"
    
    # System prompt for the LLM
    system_prompt = f"""You are a document skimming expert. Your goal is to help find specific information in a large document.

SEARCH TASK:
{search_instruction}

You will be given:
1. Document structure keywords showing where key sections are located
2. An initial overview with the first 50 lines and 25 evenly dispersed samples from the document
3. Character positions for each snippet

Use the keyword locations to intelligently navigate the document. Sections like "introduction", "conclusion", "appendix" etc. show you where major sections are.

Based on what you see, you can:
- Request to SKIP to a specific character position (e.g., "SKIP_TO_CHAR:50000")
- Request to VIEW a range around a specific character (e.g., "VIEW_CHAR_RANGE:42243:1000")
- When you find the answer, provide it as: "ANSWER_FOUND: [exact quoted text]"

You MUST respond with exactly one action per turn. Choose the action that will most efficiently find the information."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{keyword_context}\n{initial_context}"}
    ]
    
    print(f"📨 Sending initial context to DeepSeek...\n")
    
    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Get LLM response
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
            temperature=0.3  # Lower temperature for more focused responses
        )
        
        llm_response = response.choices[0].message.content
        print(f"LLM: {llm_response}")
        
        # Parse the LLM's action
        if "ANSWER_FOUND:" in llm_response:
            answer = llm_response.split("ANSWER_FOUND:")[1].strip()
            print(f"\n✓ ANSWER FOUND: {answer}")
            return answer
        
        elif "SKIP_TO_CHAR:" in llm_response:
            char_pos = int(llm_response.split("SKIP_TO_CHAR:")[1].split()[0])
            char_pos = max(0, min(char_pos, len(document_text) - 1))
            
            # Get context around this position
            context_snippet = get_text_range(document_text, char_pos, range_size=500)
            user_message = f"Showing context around character {char_pos}:\n[CHAR {char_pos}]\n{context_snippet}\n[END CONTEXT]"
            print(f"📍 Showing context around char {char_pos}")
            
        elif "VIEW_CHAR_RANGE:" in llm_response:
            parts = llm_response.split("VIEW_CHAR_RANGE:")[1].split(":")
            char_pos = int(parts[0])
            range_size = int(parts[1]) if len(parts) > 1 else 1000
            
            context_snippet = get_text_range(document_text, char_pos, range_size=range_size)
            user_message = f"Detailed view from char {char_pos-range_size} to {char_pos+range_size}:\n{context_snippet}"
            print(f"🔍 Showing detailed view around char {char_pos} (±{range_size})")
            
        else:
            print(f"⚠️ Unclear action from LLM. Asking for clarification...")
            user_message = "Please use one of these exact formats: SKIP_TO_CHAR:XXXX or VIEW_CHAR_RANGE:XXXX:YYYY or ANSWER_FOUND:[text]"
        
        # Add LLM response and our context to messages
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": user_message})
    
    print(f"\n❌ Max iterations ({max_iterations}) reached without finding answer.")
    return None

if __name__ == "__main__":
    # Example usage
    filepath = "maud_Contango_Oil_&_Gas_KKR_&_Co.txt"
    
    # Examples of different search types:
    # search_query = "What happens with Pierre Bezukhov?"  # Simple search
    # search_query = "Pierre AND marriage"                 # AND search (find sections about both)
    # search_query = "battle OR war OR conflict"           # OR search (find sections about any)
    
    search_query = "Given that Isla is granted the significant majority of board appointments, does the document outline any specific qualifications or independence requirements for these initial nine designees, or is the selection left entirely to the discretion of the respective designating parties?"
    
    result = skim_document(filepath, search_query)
    
    if result:
        print(f"\n✓ Search completed successfully!")
    else:
        print(f"\n✗ Search did not complete successfully.")
