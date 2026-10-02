import os
"""
ShiftDim: Generate Content-Based Document Shifts
Transforms documents using 8 content-based semantic shifts.
"""

import openai
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# Configuration
# ============================================================================
POE_API_KEY = os.getenv("POE_API_KEY", "")
LLM_MODEL = "llama-3.1-8b-cs"

DOCS_BASE_PATH = Path(r"c:\Coding\Code\RAG\ShiftDim\docs")
SOURCE_DOCS_PATH = Path(r"c:\Coding\Code\RAG\ShiftDim\source_docs")

# Initialize LLM client
llm_client = openai.OpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

# ============================================================================
# LLM Transformation Functions
# ============================================================================

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
# Main Processing
# ============================================================================

def generate_shifts_for_document(doc_name, text):
    """Generate all 8 shifted versions of a document in parallel."""
    
    print(f"\nGenerating shifts for {doc_name}...")
    
    # Define all shift tasks
    shift_tasks = [
        ("1_entity_relationship_extraction", "Entity and Relationship Extraction", entity_relationship_extraction),
        ("2_abstraction_normalization", "Abstraction Level Normalization", abstraction_level_normalization),
        ("3_modifier_qualifier_stripping", "Modifier and Qualifier Stripping", modifier_qualifier_stripping),
        ("4_temporal_causal_removal", "Temporal and Causal Structure Removal", temporal_causal_structure_removal),
        ("5_perspective_voice_normalization", "Perspective and Voice Normalization", perspective_voice_normalization),
        ("6_negation_isolation", "Negation Isolation", negation_isolation),
        ("7_redundancy_collapse", "Redundancy Collapse", redundancy_collapse),
        ("8_implicit_assumptions", "Implicit Assumption Extraction", implicit_assumption_extraction),
    ]
    
    shifts = {}
    
    # Run all 8 shifts in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        futures = {
            executor.submit(func, text): (key, display_name, i+1)
            for i, (key, display_name, func) in enumerate(shift_tasks)
        }
        
        # Print status for submitted tasks
        for future, (key, display_name, num) in futures.items():
            print(f"  [{num}/8] {display_name}...", end=" ", flush=True)
        print()
        
        # Wait for all to complete and collect results
        for future in as_completed(futures):
            key, display_name, num = futures[future]
            try:
                shifts[key] = future.result()
            except Exception as e:
                print(f"    Error in {display_name}: {type(e).__name__}: {e}")
                shifts[key] = "[Error generating shift]"
    
    # Print completion status
    for i, (key, display_name, func) in enumerate(shift_tasks):
        print(f"  [{i+1}/8] {display_name}... ✓")
    
    return shifts


def save_shifts_file(doc_folder, doc_name, shifts):
    """Save all shifts to DOC_N_shifts.txt file."""
    shifts_file = doc_folder / f"{doc_name}_shifts.txt"
    
    shift_names = [
        "Entity and Relationship Extraction",
        "Abstraction Level Normalization",
        "Modifier and Qualifier Stripping",
        "Temporal and Causal Structure Removal",
        "Perspective and Voice Normalization",
        "Negation Isolation",
        "Redundancy and Repetition Collapse",
        "Implicit Assumption Extraction"
    ]
    
    with open(shifts_file, 'w', encoding='utf-8') as f:
        f.write(f"DOCUMENT SHIFTS FOR {doc_name}\n")
        f.write("=" * 80 + "\n\n")
        
        for (key, content), name in zip(sorted(shifts.items()), shift_names):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"SHIFT {key.split('_')[0]}: {name}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(content)
            f.write("\n\n")
    
    print(f"    ✓ Saved to {shifts_file.name}")


def main():
    """Main processing pipeline."""
    
    print("\n" + "=" * 80)
    print("ShiftDim: Content-Based Document Shift Generation")
    print("=" * 80)
    
    docs_processed = 0
    docs_failed = 0
    
    for doc_num in range(1, 31):
        doc_name = f"DOC_{doc_num}"
        doc_file = SOURCE_DOCS_PATH / f"{doc_name}.txt"
        doc_folder = DOCS_BASE_PATH / doc_name
        
        if not doc_file.exists():
            print(f"[{doc_num}/30] {doc_name}: SKIP (file not found)")
            continue
        
        try:
            # Ensure folder exists
            doc_folder.mkdir(parents=True, exist_ok=True)
            
            # Read original document
            with open(doc_file, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            # Copy to doc folder
            dest_file = doc_folder / f"{doc_name}.txt"
            with open(dest_file, 'w', encoding='utf-8') as f:
                f.write(original_text)
            
            # Generate shifts
            shifts = generate_shifts_for_document(doc_name, original_text)
            
            # Save shifts file
            save_shifts_file(doc_folder, doc_name, shifts)
            
            print(f"[{doc_num}/30] {doc_name}: ✓ COMPLETE")
            docs_processed += 1
            
        except Exception as e:
            print(f"[{doc_num}/30] {doc_name}: ✗ FAILED ({type(e).__name__})")
            docs_failed += 1
    
    print("\n" + "=" * 80)
    print(f"Complete: {docs_processed} docs processed, {docs_failed} failed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
