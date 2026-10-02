# Document Analysis Agent

A multi-agent document analysis system that uses DeepSeek LLM with function calling to intelligently search, analyze, and answer questions from a collection of text documents.

## Features

- **14 Specialized Tools**: Search, semantic search, focus management, entity extraction, document comparison, and more
- **Smart Search**: Multiple modes (simple, boolean, fuzzy, exact) with auto-fallback
- **Semantic Search**: Powered by Ollama embeddings (nomic-embed-text)
- **Focus Management**: Maintain a working set of relevant documents
- **Persistent Storage**: Cached embeddings and focus list across sessions
- **Interactive Mode**: Chat-like interface for asking questions

## Setup

### Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running locally
   - Install from: https://ollama.ai
   - Pull the embedding model: `ollama pull nomic-embed-text`
3. **DeepSeek API Key** (already configured in config.py)

### Installation

```bash
# Navigate to the doc_agent directory
cd doc_agent

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running (in a separate terminal)
ollama serve
```

## Usage

### Interactive Mode

```bash
python agent.py
```

This starts an interactive session where you can ask questions about your documents.

### Example Queries

- "What are the main topics discussed in these documents?"
- "Find all documents mentioning confidentiality agreements"
- "What dates are mentioned in the focused documents?"
- "Compare the documents that mention non-disclosure"
- "Extract all dollar amounts from the documents"

## System Architecture

### Components

1. **Document Manager** (`utils/document_manager.py`)
   - Loads all .txt files from the data directory
   - Intelligent chunking (hierarchical: paragraphs → sentences → characters)
   - Generates and caches embeddings using Ollama
   - Manages document metadata

2. **Focus Manager** (`utils/focus_manager.py`)
   - Maintains a "documents of interest" list
   - Persists focus list to `data/focus_docs.json`
   - Tracks reasons for focusing on documents

3. **Search Utilities** (`utils/search_utils.py`)
   - Keyword search (simple, boolean, fuzzy, exact)
   - Semantic search with cosine similarity
   - Snippet extraction for context

4. **Agent Tools** (`tools/agent_tools.py`)
   - 14 distinct tools with JSON schemas
   - Function calling integration
   - Error handling and validation

5. **Agent Orchestrator** (`agent.py`)
   - Main loop managing LLM interactions
   - Tool execution and result handling
   - Conversation history management
   - Max 20 iterations per query

### Tool Catalog

1. **search** - Keyword search with multiple modes
2. **semantic_search** - Embedding-based similarity search
3. **list_docs** - List available documents
4. **add_to_focus** - Add documents to focus list
5. **remove_from_focus** - Remove documents using filters or IDs
6. **list_focused_docs** - Show current focus list
7. **clear_focus** - Clear all focused documents
8. **read_doc** - Read document content (preview/full/chunk)
9. **get_metadata** - Get document statistics
10. **get_context** - Find keyword mentions with context
11. **extract_entities** - Extract dates, money, legal terms
12. **compare_docs** - Compare documents for similarities
13. **ask_user** - Request clarification
14. **submit_answer** - Submit final answer (terminal)

## Workflow Example

For query: "What confidentiality terms are mentioned?"

1. Agent calls `semantic_search(query="confidentiality terms")`
2. Adds relevant documents to focus with `add_to_focus()`
3. Calls `extract_entities(scope="focused", entity_types=["legal"])`
4. Reads specific sections with `read_doc(mode="chunk")`
5. Submits answer with `submit_answer(answer=..., sources=[...])`

## Configuration

Edit `config.py` to customize:

- API keys and endpoints
- Embedding model and parameters
- Search thresholds
- Agent behavior (max iterations, history limit)
- File paths

## Data Structure

```
doc_agent/
├── agent.py                 # Main orchestrator
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── tools/
│   └── agent_tools.py     # All 14 tools
├── utils/
│   ├── document_manager.py
│   ├── focus_manager.py
│   └── search_utils.py
└── data/
    ├── *.txt              # Source documents (already exists)
    ├── embeddings.pkl     # Cached embeddings (auto-generated)
    └── focus_docs.json    # Persisted focus list (auto-generated)
```

## Performance

- **First run**: Generates embeddings (may take 5-10 minutes for many documents)
- **Subsequent runs**: Instant load from cache
- **Query response**: Typically 3-8 tool calls, 10-30 seconds per query

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Verify model is available: `ollama list`
- Pull model if needed: `ollama pull nomic-embed-text`

### DeepSeek API Error
- Check API key in `config.py`
- Verify internet connection
- Check API rate limits

### Memory Issues
- Reduce `MAX_EMBEDDING_TOKENS` in config.py
- Process fewer documents at once
- Clear embeddings cache and regenerate

## Advanced Usage

### Programmatic Usage

```python
from agent import DocumentAnalysisAgent

# Initialize agent
agent = DocumentAnalysisAgent()
agent.initialize()

# Run a query
result = agent.run_query("What are the main topics?")

# Access results
print(result['answer'])
print(result['sources'])
print(result['confidence'])
```

### Custom Tools

Add new tools in `tools/agent_tools.py`:

1. Implement tool function
2. Add to `TOOL_FUNCTIONS` dictionary
3. Add schema to `TOOL_DEFINITIONS`

## License

MIT License - See LICENSE file for details
