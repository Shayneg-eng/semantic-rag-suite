# Quick Start Guide - Document Analysis Agent

## Prerequisites Check

1. ✅ Python 3.8+ installed
2. ✅ Ollama installed and running
3. ✅ Documents in `../data/*.txt` directory

## Step-by-Step Setup

### 1. Install Dependencies

```bash
cd doc_agent
pip install -r requirements.txt
```

### 2. Start Ollama (if not running)

In a separate terminal:
```bash
ollama serve
```

Then pull the embedding model:
```bash
ollama pull nomic-embed-text
```

### 3. Test the System

Run the test script to verify everything works:
```bash
python test_agent.py
```

Choose option 1 for a quick test (no LLM calls).

### 4. Run the Agent

Start interactive mode:
```bash
python agent.py
```

## Example Session

```
📝 Your question: What types of agreements are in these documents?

[Agent will search, analyze, and respond...]

FINAL ANSWER
============================================================

The documents contain primarily Non-Disclosure Agreements (NDAs) and 
confidentiality agreements. Common themes include:

1. Mutual Non-Disclosure Agreements
2. Client NDAs
3. Supplier Confidentiality Agreements
4. Data Use Agreements

Sources: contractnli_BCG-Mutual-NDA.txt, contractnli_BT_NDA.txt, ...
Confidence: high
============================================================
```

## Common Issues & Solutions

### Issue: "Connection refused" when embedding

**Solution**: Start Ollama service
```bash
ollama serve
```

### Issue: "Model not found: nomic-embed-text"

**Solution**: Pull the model
```bash
ollama pull nomic-embed-text
```

### Issue: Slow first run

**Explanation**: Generating embeddings for all documents (one-time process)
- Cached in `data/embeddings.pkl`
- Subsequent runs load instantly from cache

### Issue: API error from DeepSeek

**Solution**: Check API key in `config.py` and internet connection

## Understanding the Workflow

The agent follows this strategy:

1. **Discover**: Broad search to find relevant documents
   - Uses `search()` or `semantic_search()`

2. **Focus**: Add promising documents to working set
   - Uses `add_to_focus()`

3. **Refine**: Remove irrelevant documents
   - Uses `remove_from_focus()` with filters

4. **Analyze**: Deep dive into focused documents
   - Uses `read_doc()`, `extract_entities()`, etc.

5. **Answer**: Submit final response
   - Uses `submit_answer()`

## Tips for Best Results

1. **Be specific**: "What confidentiality clauses..." vs "Tell me about these"
2. **Use natural language**: The agent understands conversational queries
3. **Check sources**: Always review the source documents cited
4. **Iterative refinement**: Ask follow-up questions to dig deeper

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Customize `config.py` for your needs
- Add your own documents to `../data/` directory
- Experiment with different query types

## Need Help?

- Check the agent's output for debugging info
- Review the 14 available tools in `tools/agent_tools.py`
- Look at example workflows in the main specification
