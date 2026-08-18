# rag-practice

A hands-on repository for learning **Retrieval-Augmented Generation (RAG)** from first principles, then evolving the same system into **Agentic RAG**.

## Learning path

1. **RAG foundations** — loading, chunking, embeddings, vector search, prompt construction.
2. **Retrieval quality** — chunk-size experiments, top-k, metadata, hybrid search, reranking.
3. **Evaluation** — measure retrieval before judging generation.
4. **Agentic RAG** — query rewriting, routing, document grading, retry loops and tools.

The first milestone intentionally avoids LangChain/LlamaIndex so the core mechanics stay visible.

## Architecture: milestone 1

```text
Documents
   ↓
Loader
   ↓
Chunker
   ↓
SentenceTransformer embeddings
   ↓
Qdrant (local, on disk)
   ↓
Top-k dense retrieval
   ↓
Prompt with retrieved context
   ↓
Ollama
   ↓
Answer + retrieved sources
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally
- An Ollama chat model, e.g. `ollama pull llama3.2`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
ollama pull llama3.2
```

Put `.pdf`, `.txt`, or `.md` files under `data/raw/`.

## Run

Index the documents:

```bash
rag-practice index data/raw
```

Ask a question:

```bash
rag-practice ask "What is retrieval-augmented generation?"
```

Use another Ollama model:

```bash
rag-practice ask "What is RAG?" --model qwen3:8b
```

The CLI prints the retrieved chunks before the final answer on purpose. While learning RAG, inspect retrieval first: if the relevant evidence is missing, generation cannot reliably fix it.

## Repository layout

```text
src/rag_practice/
├── chunking.py     # deliberately simple chunking
├── documents.py    # PDF/text/Markdown loading
├── embeddings.py   # local embedding model
├── store.py        # Qdrant indexing + retrieval
├── generation.py   # context → prompt → LLM
└── cli.py          # index / ask commands
```

## Next milestones

- [ ] Build a small evaluation dataset
- [ ] Measure Recall@k / MRR
- [ ] Compare chunk sizes and overlaps
- [ ] Add sparse/BM25 retrieval
- [ ] Add hybrid retrieval + RRF
- [ ] Add a reranker
- [ ] Add query rewriting
- [ ] Convert retrieval into tools
- [ ] Add LangGraph state + routing
- [ ] Add document grading and retry loops

## Principle

Keep this distinction clear throughout the project:

```text
Agent layer: planning, routing, tool calling, retry/reflection
                         ↓ uses
RAG layer: chunking, embedding, retrieval, reranking, generation
```

An agent does not replace RAG; it orchestrates RAG and other tools.
