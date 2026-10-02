# Semantic RAG Suite

A research playground for **retrieval-augmented generation**: a collection of retrieval and
re-ranking strategies, benchmarked against each other on a document corpus.

## Core scripts
| File | Role |
|---|---|
| `generate_embeddings.py`, `embed_documents_fast.py`, `embed_chunks_parallel.py` | Build embeddings |
| `search_documents.py`, `test_chunk_search.py` | Retrieval |
| `hybrid_scoring.py`, `tune_weights.py`, `tune_percentiles.py` | Hybrid lexical+semantic scoring |
| `run_benchmark.py`, `compare_methods.py`, `final_comparison.py`, `evaluate_ranking.py` | Benchmark harness |
| `visualize_search.py` | Visualize results |

## Experimental variants
`ShiftDim*`, `SplitDim`, `SieveDim`, `Skimming`, `AgentSearch`/`AgentSearch2` each explore a
different dimension-reduction / retrieval idea (agentic search, dimension sieving, skimming).

## Data
The document corpus (~22k files) and generated embeddings are **not committed** — see
[`DATA.md`](DATA.md). Keys are read from the environment (`.env.example`).

## Run it
```bash
python -m pip install openai numpy scikit-learn
export DEEPSEEK_API_KEY="sk-..."
python generate_embeddings.py   # point it at your corpus first
python run_benchmark.py
```
