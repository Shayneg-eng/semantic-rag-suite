# Usage Examples - Document Analysis Agent

## Example 1: Simple Question Answering

**Query**: "What are the main types of agreements in these documents?"

**Agent Workflow**:
```
1. semantic_search(query="types of agreements")
   → Finds relevant chunks across documents
   
2. add_to_focus([found document IDs])
   → Adds 5 most relevant documents to focus
   
3. extract_entities(scope="focused", entity_types=["legal"])
   → Extracts legal terms from focused documents
   
4. get_context(keywords=["agreement", "contract", "NDA"])
   → Gets context around key terms
   
5. submit_answer(
     answer="The documents primarily contain Non-Disclosure Agreements (NDAs)...",
     source_doc_ids=[...],
     confidence="high"
   )
```

**Expected Output**:
```
FINAL ANSWER
============================================================
The documents primarily contain Non-Disclosure Agreements (NDAs) 
and confidentiality agreements. Main types include:

1. Mutual Non-Disclosure Agreements (MNDA)
2. Unilateral NDAs
3. Data Use Agreements
4. Client Confidentiality Agreements
5. Supplier NDAs

Sources: contractnli_BCG-Mutual-NDA.txt, contractnli_BT_NDA.txt, ...
Confidence: high
============================================================
```

---

## Example 2: Finding Specific Information

**Query**: "What monetary amounts are mentioned in the documents?"

**Agent Workflow**:
```
1. list_docs(scope="all")
   → See what documents are available
   
2. extract_entities(scope="all", entity_types=["money"])
   → Extract all dollar amounts
   
3. add_to_focus([documents with money mentioned])
   → Focus on documents with monetary values
   
4. read_doc(doc_id="...", mode="preview")
   → Read specific documents for context
   
5. submit_answer(
     answer="Found monetary amounts in X documents...",
     source_doc_ids=[...],
     excerpts=["$1,000,000 maximum liability", ...]
   )
```

---

## Example 3: Comparative Analysis

**Query**: "Compare the confidentiality terms in the first three documents"

**Agent Workflow**:
```
1. list_docs(scope="all")
   → Get list of all documents
   
2. add_to_focus([first_3_doc_ids])
   → Add first 3 documents to focus
   
3. get_context(keywords=["confidential", "disclosure", "obligation"], scope="focused")
   → Find how these terms are used
   
4. compare_docs(scope="focused")
   → Find common and unique terms
   
5. read_doc(doc_id="...", mode="chunk", chunk_id=X)
   → Read specific sections
   
6. submit_answer(
     answer="Common terms: All three require non-disclosure...",
     source_doc_ids=[...],
     confidence="high"
   )
```

---

## Example 4: Focus Management

**Query**: "Find all documents about mutual agreements, then analyze their effective dates"

**Agent Workflow**:
```
1. search(query="mutual", mode="exact", scope="all")
   → Finds documents with exact "mutual" match
   → Auto-adds to focus list
   
2. list_focused_docs()
   → Shows what's in focus (6 documents)
   
3. remove_from_focus(filter_query="documents without dates")
   → Removes documents that don't mention dates
   
4. extract_entities(scope="focused", entity_types=["dates"])
   → Extracts dates from remaining focused documents
   
5. submit_answer(...)
```

---

## Example 5: Deep Dive Investigation

**Query**: "What are the liability clauses in the Bosch NDA?"

**Agent Workflow**:
```
1. search(query="Bosch", mode="simple", scope="all")
   → Finds the Bosch document
   
2. add_to_focus(["contractnli_01_Bosch-Automotive..."])
   
3. semantic_search(query="liability indemnification limitation", scope="focused")
   → Finds relevant sections
   
4. read_doc(doc_id="...", mode="chunk", chunk_id=X)
   → Reads the specific liability section
   
5. get_context(keywords=["liability", "indemnify", "limitation"], scope="focused")
   → Gets surrounding context
   
6. submit_answer(
     answer="The Bosch NDA includes the following liability provisions...",
     excerpts=["Neither party shall be liable for...", ...],
     confidence="high"
   )
```

---

## Example 6: Filtering and Refinement

**Query**: "Which documents are shorter than 2000 words and mention 'data'?"

**Agent Workflow**:
```
1. get_metadata(scope="all")
   → Gets word counts for all documents
   
2. search(query="data", mode="simple", scope="all")
   → Finds documents mentioning "data"
   
3. add_to_focus([documents from search])
   
4. remove_from_focus(filter_query="documents longer than 2000 words")
   → Filters by length
   
5. list_focused_docs()
   → Shows final filtered list
   
6. submit_answer(
     answer="Found X documents matching criteria...",
     source_doc_ids=[...]
   )
```

---

## Example 7: Entity Extraction

**Query**: "Extract all dates and monetary amounts from documents about data use"

**Agent Workflow**:
```
1. search(query="data use", mode="fuzzy", scope="all")
   → Fuzzy search finds "Data Use Agreement" docs
   
2. add_to_focus([found_docs])
   
3. extract_entities(scope="focused", entity_types=["dates", "money"])
   → Extracts structured data
   
4. submit_answer(
     answer="Found the following dates and amounts...",
     excerpts=["Effective Date: January 1, 2019", "$10,000 penalty", ...]
   )
```

---

## Example 8: Boolean Search

**Query**: "Find documents that mention 'patent' OR 'trademark' but NOT 'copyright'"

**Agent Workflow**:
```
1. search(query="patent OR trademark NOT copyright", mode="boolean", scope="all")
   → Boolean logic applied
   
2. add_to_focus([matching_docs])
   
3. get_context(keywords=["patent", "trademark"], scope="focused")
   → Shows usage context
   
4. submit_answer(
     answer="Found X documents with intellectual property terms...",
     source_doc_ids=[...]
   )
```

---

## Example 9: Clarification Request

**Query**: "Analyze the agreement" (ambiguous)

**Agent Workflow**:
```
1. list_docs(scope="all")
   → Sees there are 50+ agreements
   
2. ask_user(question="Which specific agreement would you like me to analyze? Or should I analyze all agreements for common patterns?")
   → Agent requests clarification
   
[User responds: "All agreements"]

3. get_metadata(scope="all")
   → Gets overview statistics
   
4. extract_entities(scope="all", entity_types=["legal"])
   → Finds common legal terms
   
5. compare_docs(doc_ids=[sample of 5 docs])
   → Compares for patterns
   
6. submit_answer(
     answer="Analysis of all agreements shows...",
     confidence="medium"
   )
```

---

## Example 10: Multi-Step Complex Query

**Query**: "Which documents have confidentiality periods longer than 2 years, and what are those specific periods?"

**Agent Workflow**:
```
1. semantic_search(query="confidentiality period duration term years")
   → Finds relevant sections
   
2. add_to_focus([docs from semantic search])
   
3. extract_entities(scope="focused", entity_types=["dates"])
   → Extracts date-related information
   
4. get_context(keywords=["year", "years", "period", "term"], scope="focused")
   → Gets context around duration mentions
   
5. For each focused document:
   read_doc(doc_id=doc, mode="chunk", chunk_id=relevant_chunk)
   → Reads specific sections about periods
   
6. remove_from_focus(filter_query="documents without year mentions")
   → Filters out documents without clear periods
   
7. submit_answer(
     answer="Documents with 2+ year confidentiality periods:\n1. Doc A: 3 years\n2. Doc B: 5 years...",
     source_doc_ids=[...],
     excerpts=["...confidential for a period of three (3) years...", ...],
     confidence="medium"
   )
```

---

## Tips for Effective Queries

### Good Queries
- ✅ "What are the termination clauses in the mutual NDAs?"
- ✅ "Compare liability limits across all agreements"
- ✅ "Extract all company names mentioned in documents"
- ✅ "Which documents were effective in 2019?"

### Queries Needing Refinement
- ❌ "Analyze this" → Too vague, agent will ask for clarification
- ❌ "Everything about agreements" → Too broad, better to specify aspect
- ⚠️  "Tell me about the document" → Which document? Be specific

### Query Types Best Suited for Each Tool

**For keyword search**: 
- "Find documents containing 'termination'"
- "Which docs mention both 'patent' AND 'license'?"

**For semantic search**:
- "What are the obligations of the receiving party?"
- "How is confidential information defined?"

**For entity extraction**:
- "What dates are mentioned?"
- "Extract all monetary values"

**For comparison**:
- "What's common across these 3 documents?"
- "How do the NDAs differ?"

---

## Interactive Session Example

```bash
$ python agent.py

============================================================
Document Analysis Agent - Interactive Mode
============================================================

Type your questions, or 'quit' to exit

📝 Your question: What types of NDAs are in the collection?

--- Iteration 1/20 ---
🔧 Calling tool: semantic_search
   Arguments: {"query": "types of NDAs non-disclosure agreements", "scope": "all", "top_k": 10}
   
--- Iteration 2/20 ---
🔧 Calling tool: add_to_focus
   Arguments: {"doc_ids": ["contractnli_BCG-Mutual-NDA.txt", ...]}
   
--- Iteration 3/20 ---
🔧 Calling tool: extract_entities
   Arguments: {"scope": "focused", "entity_types": ["legal"]}

--- Iteration 4/20 ---
🔧 Calling tool: submit_answer
   Arguments: {"answer": "The collection contains...", "source_doc_ids": [...]}

✅ Final answer submitted!

============================================================
FINAL ANSWER
============================================================

The collection contains several types of Non-Disclosure Agreements:

1. **Mutual NDAs**: Where both parties exchange confidential information
   - Example: BCG Mutual NDA, DoiT ICN NDA
   
2. **Unilateral NDAs**: One-way information disclosure
   - Example: BT NDA, NSK Supplier Agreement
   
3. **Data Use Agreements**: Specific to data sharing
   - Example: NYC Data Use Agreement, MDCH Data Use Agreement
   
4. **Industry-Specific NDAs**: 
   - Healthcare (HNBA, APIC)
   - Technology (FullStory, ePSTEEN)
   - Automotive (Bosch, NSK)

Sources: contractnli_BCG-Mutual-NDA.txt, contractnli_BT_NDA.txt, ...
Confidence: high
============================================================

📝 Your question: quit

Goodbye!
```

---

## Advanced Patterns

### Pattern 1: Iterative Refinement
```
Query 1: "Find all confidentiality agreements"
Query 2: "From those, which mention data?"
Query 3: "What are the data protection clauses in those documents?"
```

### Pattern 2: Comparative Analysis
```
Query: "Compare documents A, B, and C for differences in termination clauses"
→ Agent will focus on those 3, extract relevant sections, and compare
```

### Pattern 3: Aggregate Statistics
```
Query: "What's the average length of agreements in the collection?"
→ Agent uses get_metadata to calculate
```

### Pattern 4: Targeted Extraction
```
Query: "Find all documents with effective dates in 2017, list those dates"
→ Agent combines search + entity extraction + filtering
```
