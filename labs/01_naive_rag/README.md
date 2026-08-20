# Lab 01 — Naive RAG From Scratch

## Learning objective

Build the smallest inspectable end-to-end RAG pipeline without an orchestration framework:

```text
documents → fixed chunks → embeddings → in-memory vector index
          → top-k retrieval → context/prompt → generator → answer + citations
```

The lab deliberately uses a deterministic `HashingEmbedder` and an extractive generator. They keep the mechanics reproducible and make failure attribution obvious. They are **not** substitutes for a semantic neural embedder or an LLM; those enter later milestones as explicit variables.

## Run

```bash
python labs/01_naive_rag/run.py
pytest
```

The experiment reuses the M00 corpus and writes its result artifact to:

- `results/baseline.json` — machine-readable metrics and query traces
- `results/baseline.md` — baseline comparison, failure analysis, findings

## Evaluation contract

**Hypothesis:** retrieval should make the context-only generator answerable on questions where evidence is retrieved, while extractive generation should remain grounded.

**Retrieval baseline:** M00 BM25 on the same M01 questions.

**Generation baseline:** the same extractive generator with no retrieved context.

**Retrieval metrics:** MRR, MAP, Hit Rate, Precision, Recall, nDCG.

**Generation metrics:** reference containment, token F1, grounded token recall, citation precision/recall.

**System metrics:** embedding/retrieval/generation/end-to-end latency and approximate prompt/output token counts.

## Important limitation

`HashingEmbedder` hashes lexical features into dense vectors. Dense storage does not make the representation semantic. The preserved paraphrase failure is intentional and becomes a target for M02.
