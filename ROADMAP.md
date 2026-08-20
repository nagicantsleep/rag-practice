# RAG Practice — Learning Roadmap

This file is the source of truth for the learning sequence and completion status in this repository.

## Goal

Learn Retrieval-Augmented Generation by implementing mechanisms directly, evaluating them on shared benchmarks, comparing them with explicit baselines, and only then introducing higher-level frameworks or production abstractions.

The repository should answer more than “does it run?”: why a method works or fails, which component changed quality, what the latency/token/index trade-offs are, and whether conclusions are reproducible.

## Non-negotiable learning contract

1. **Evaluation is mandatory.** A demo or working pipeline without a baseline, benchmark, metrics, saved results, and error analysis is incomplete.
2. **Baseline before improvement.** New methods must be compared with the simplest relevant prior implementation.
3. **Shared benchmarks where tasks overlap.** Do not change both method and benchmark and then attribute the difference to the method.
4. **Mechanisms before frameworks.** Early labs keep retrieval/ranking/control logic visible instead of hiding it behind orchestration frameworks.
5. **Observable pipelines.** Trace transformed queries, retrieval scores, selected context, answer, citations, metrics, latency, tokens, and cost where applicable.
6. **Reproducibility.** Record dataset/model versions, chunking, retrieval parameters, seeds where applicable, and evaluator configuration.

## Evaluation contract for every lab

Before a lab can be `DONE`, it must define and save:

- **Hypothesis** — what should improve and why.
- **Baseline** — at least one comparable prior implementation.
- **Dataset/benchmark** — corpus, queries/questions, relevance labels or reference answers, splits/provenance where applicable.
- **Retrieval metrics** — e.g. Hit Rate@K, Precision@K, Recall@K, MRR, MAP, nDCG@K.
- **Generation/RAG metrics** — e.g. correctness, groundedness, context relevance, citation precision/recall, refusal/unsupported-answer behavior.
- **System metrics** — relevant latency, tokens, API cost, index build time/size, memory footprint.
- **Error analysis** — representative failures classified as query, retrieval, ranking, context, evidence, generation, citation, or control-flow failures.
- **Saved artifacts** — machine-readable results plus a human-readable findings report.

A key rule: **retrieval quality must be evaluated independently from final generation whenever retrieval is involved.** A language model must not be allowed to hide retrieval failures.

## Definition of Done

- [ ] learning objective written
- [ ] algorithm/pipeline implemented
- [ ] core behavior covered by automated tests
- [ ] baseline identified and runnable
- [ ] benchmark/evaluation dataset defined
- [ ] retrieval evaluation present when retrieval is involved
- [ ] generation/groundedness evaluation present when generation is involved
- [ ] relevant system metrics recorded
- [ ] configuration reproducible
- [ ] results saved, not only printed
- [ ] representative failures inspected
- [ ] findings/trade-offs written down

## Milestones

Statuses: `TODO`, `IN PROGRESS`, `DONE`.

### M00 — Information Retrieval Fundamentals — `DONE`

Implemented:

- minimal tokenization and inverted index
- TF / IDF / TF-IDF
- dense and sparse cosine similarity
- BM25 from the scoring formula
- top-k retrieval
- relevance judgments
- Hit Rate@K / Precision@K / Recall@K / MRR / MAP / nDCG@K
- deterministic benchmark and hand-checkable metric tests
- TF-IDF vs BM25 comparison
- query-level error analysis

Artifacts: `src/rag_practice/ir/`, `src/rag_practice/evaluation/`, `benchmarks/m00_ir/`, `labs/00_ir_fundamentals/`.

Key finding: lexical retrieval fails a vocabulary-mismatch paraphrase. That failure is retained as a regression target.

### M01 — Naive RAG From Scratch — `DONE`

Pipeline:

```text
documents → fixed chunks → embeddings → in-memory vector index
          → top-k retrieval → context/prompt → generator → answer + citations
```

Implemented:

- `Document`, `Chunk`, retrieval result, generated answer, and trace models
- fixed-size chunking with overlap
- swappable embedding interface
- deterministic hashing-based dense vector representation
- minimal in-memory cosine vector index reusing the M00 cosine primitive
- generator interface plus deterministic extractive generator
- context/prompt construction
- chunk citations
- embedding/retrieval/generation/end-to-end trace timing
- approximate prompt/output token counts
- deterministic answer correctness, groundedness, and citation metrics
- M01 QA benchmark reusing the M00 corpus
- BM25 retrieval baseline and no-retrieval generation baseline

Evaluation summary:

- hashing-vector Hit Rate@1: **0.833**
- BM25 Hit Rate@1: **0.833**
- hashing-vector Recall@3: **0.833**
- BM25 Recall@3: **1.000**
- answer contains reference: **0.833** vs no-retrieval **0.000**
- grounded token recall: **1.000**
- citation precision/recall: **0.833 / 0.833**

Key finding: M01 exposes a clean **grounded-but-wrong** case. The extractive answer is fully supported by the retrieved chunk, but retrieval selected the wrong evidence for the hard paraphrase query. Dense vector storage alone does not create semantic understanding; the embedding representation determines semantic capability.

Artifacts: `src/rag_practice/core/`, `src/rag_practice/embeddings/`, `src/rag_practice/retrieval/`, `src/rag_practice/generation/`, `src/rag_practice/rag/`, `benchmarks/m01_rag/`, `labs/01_naive_rag/`.

### M02 — Retrieval Families — `TODO`

Implement and compare on shared query classes:

- BM25 / sparse lexical retrieval
- real neural semantic dense retrieval
- current hashing-vector mechanics baseline
- hybrid sparse + dense
- Reciprocal Rank Fusion (RRF)
- learned sparse retrieval such as SPLADE as an advanced sub-lab
- late-interaction retrieval such as ColBERT/ColBERTv2 as an advanced sub-lab

Evaluation focus: Recall@K, MRR, nDCG@K, exact/entity vs semantic/paraphrase query classes, latency, index size. First acceptance target: test whether a neural dense embedder fixes the preserved paraphrase failure without regressing exact-term queries.

### M03 — Indexing and Chunking — `TODO`

Compare fixed chunks, overlap, sentence/paragraph-aware chunks, semantic chunking, metadata enrichment, parent-child retrieval, and hierarchical indexes.

Evaluation: retrieval vs granularity, evidence completeness, redundancy, context-token utilization.

### M04 — Reranking and Context Construction — `TODO`

Implement retrieve-many/rerank-few, cross-encoder reranking, LLM reranking, MMR, redundancy control, and context ordering/packing.

Evaluation: ranking metrics before/after, answer quality at fixed context budgets, latency-quality trade-off.

### M05 — Query Transformation — `TODO`

Implement query rewriting, multi-query retrieval, RAG-Fusion, Query2Doc-style expansion, HyDE, and query decomposition.

Evaluation: original vs transformed-query retrieval, per-query-class wins/losses, transformation cost.

### M06 — Multi-hop, Active, Adaptive, and Self-correcting RAG — `TODO`

Implement progressively: multi-hop/iterative retrieval, FLARE-style active retrieval, no-RAG/single/iterative routing, Adaptive-RAG concepts, Corrective RAG, and Self-RAG-style retrieve/critique/reflection control.

Evaluation: simple vs multi-hop subsets, unnecessary-retrieval rate, correction success, unsupported-answer rate, loop cost/latency.

### M07 — Hierarchical, Graph, and Memory-oriented RAG — `TODO`

Implement progressively: RAPTOR-style trees, knowledge-graph retrieval fundamentals, GraphRAG local/global patterns, LightRAG ideas, KAG-style structured reasoning, HippoRAG-style associative retrieval, and memory-oriented patterns.

Evaluation: local vs global questions, relation/multi-hop questions, flat-vector baselines, construction/update cost.

### M08 — Specialized Sources and Modalities — `TODO`

Sub-labs: Web RAG, SQL/structured RAG, metadata/filter-aware RAG, Code RAG, multimodal RAG, visual-document/page-image RAG, and long-context vs retrieval routing.

Evaluation remains source/modality appropriate while retaining grounding, latency, and cost measurements where applicable.

### M09 — Agentic RAG — `TODO`

Implement planner, search strategy, tool/source router, retrieval loop, evidence evaluator, retry/stop policy, memory/state, then multi-agent variants.

Evaluation: task success, tool/retrieval precision, steps/tool calls, unnecessary actions, recovery, grounding, latency, cost.

### M10 — Training and Production RAG — `TODO`

Study/implement selectively: retriever fine-tuning, hard-negative mining, learned rerankers, end-to-end RAG concepts, caching, incremental indexing, observability, permissions, prompt-injection/adversarial-retrieval defenses, freshness policies, scaling/serving.

Evaluation: offline quality, online/system performance, robustness/security, freshness, regression testing.

## Immediate next step

Start **M02 — Retrieval Families** with a real neural semantic dense retriever, keep BM25 and hashing-vector baselines, and use the preserved paraphrase failure as the first explicit acceptance test.
