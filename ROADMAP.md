# RAG Practice — Learning Roadmap

This file is the source of truth for the learning sequence and completion status in this repository.

## Goal

Learn Retrieval-Augmented Generation by implementing mechanisms directly, evaluating them on shared benchmarks, comparing them with explicit baselines, and only then introducing higher-level frameworks or production abstractions.

## Non-negotiable learning contract

1. **Evaluation is mandatory.** A demo or working pipeline without a baseline, benchmark, metrics, saved results, and error analysis is incomplete.
2. **Baseline before improvement.** New methods must be compared with the simplest relevant prior implementation.
3. **Shared benchmarks where tasks overlap.** Do not change both method and benchmark and then attribute the difference to the method.
4. **Mechanisms before frameworks.** Early labs keep retrieval/ranking/control logic visible instead of hiding it behind orchestration frameworks.
5. **Observable pipelines.** Trace transformed queries, retrieval scores, selected context, answer, citations, metrics, latency, tokens, and cost where applicable.
6. **Reproducibility.** Record dataset/model versions, chunking, retrieval parameters, seeds where applicable, and evaluator configuration.

## Evaluation contract for every lab

Before a lab can be `DONE`, it must define and save hypothesis, baseline, benchmark, relevant retrieval/generation/system metrics, error analysis, and machine-readable plus human-readable result artifacts.

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

Implemented minimal tokenization/inverted index, TF-IDF, dense/sparse cosine, BM25, top-k retrieval, relevance judgments, Hit Rate/Precision/Recall/MRR/MAP/nDCG, deterministic benchmark, hand-checkable metric tests, baseline comparison, and failure analysis.

Key finding: lexical retrieval fails a vocabulary-mismatch paraphrase. That failure is retained as a regression target.

### M01 — Naive RAG From Scratch — `DONE`

Implemented the inspectable pipeline:

```text
documents → fixed chunks → embeddings → in-memory vector index
          → top-k retrieval → context/prompt → generator → answer + citations
```

Includes data models, fixed-size chunking, hashing embeddings, vector index, generator abstraction, extractive generator, context/prompt construction, citations, tracing, retrieval evaluation, answer/grounding/citation evaluation, latency/token measurements, BM25 baseline, and no-retrieval baseline.

Key finding: a RAG answer can be **fully grounded but wrong** when retrieval selects the wrong evidence.

### M02 — Retrieval Families — `IN PROGRESS`

#### Core sub-lab — `DONE`

Implemented and evaluated:

- BM25 lexical baseline
- hashing-vector dense-storage baseline
- supervised neural dual encoder trained from scratch
- Reciprocal Rank Fusion (RRF)
- min-max normalized weighted sparse+dense fusion
- train/dev/test separation for hybrid tuning
- exact-vs-semantic query-class evaluation

Held-out summary:

| Method | Recall@1 | Exact R@1 | Semantic R@1 | Recall@3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.800 | 1.000 | 0.600 | 0.900 |
| Hashing | 0.800 | 1.000 | 0.600 | 0.800 |
| Neural dual encoder | 0.800 | 0.800 | 0.800 | 1.000 |
| Hybrid RRF | 0.800 | 0.800 | 0.800 | 1.000 |
| Hybrid weighted | **0.900** | **1.000** | **0.800** | **1.000** |

The preserved paraphrase `conceptual likeness between paraphrases` is retrieved correctly by the learned neural encoder and weighted hybrid, while BM25 and hashing miss it.

Core finding: **learned representation** creates semantic matching; vector storage by itself does not. Dense retrieval also regresses some exact queries, making sparse+dense complementarity measurable rather than assumed.

Artifacts: `src/rag_practice/retrieval/neural_dual_encoder.py`, `src/rag_practice/retrieval/fusion.py`, `benchmarks/m02_retrieval/`, `labs/02_retrieval_families/`.

#### Remaining M02 sub-labs

- [ ] pretrained general-purpose semantic embedding baseline
- [ ] learned sparse retrieval (SPLADE family)
- [ ] late interaction (ColBERT/ColBERTv2)
- [ ] larger benchmark / index-size and latency comparison

M02 remains `IN PROGRESS` until the remaining families are implemented or explicitly scoped with evidence.

### M03 — Indexing and Chunking — `TODO`

Compare fixed chunks, overlap, sentence/paragraph-aware chunks, semantic chunking, metadata enrichment, parent-child retrieval, and hierarchical indexes. Evaluate retrieval vs granularity, evidence completeness, redundancy, and token-budget utilization.

### M04 — Reranking and Context Construction — `TODO`

Implement retrieve-many/rerank-few, cross-encoder reranking, LLM reranking, MMR, redundancy control, and context ordering/packing. Evaluate ranking changes, answer quality at fixed context budgets, and latency-quality trade-offs.

### M05 — Query Transformation — `TODO`

Implement query rewriting, multi-query retrieval, RAG-Fusion, Query2Doc-style expansion, HyDE, and query decomposition. Evaluate original vs transformed retrieval, query-class wins/losses, and transformation cost.

### M06 — Multi-hop, Active, Adaptive, and Self-correcting RAG — `TODO`

Implement multi-hop/iterative retrieval, FLARE-style active retrieval, no-RAG/single/iterative routing, Adaptive-RAG concepts, Corrective RAG, and Self-RAG-style retrieve/critique/reflection control. Evaluate complexity routing, correction, unsupported-answer rate, and loop cost.

### M07 — Hierarchical, Graph, and Memory-oriented RAG — `TODO`

Implement RAPTOR-style trees, knowledge-graph retrieval, GraphRAG local/global patterns, LightRAG ideas, KAG-style structured reasoning, HippoRAG-style associative retrieval, and memory-oriented retrieval. Evaluate local/global and multi-hop relation questions plus construction/update cost.

### M08 — Specialized Sources and Modalities — `TODO`

Sub-labs: Web RAG, SQL/structured RAG, metadata/filter-aware RAG, Code RAG, multimodal RAG, visual-document/page-image RAG, and long-context vs retrieval routing, each with source-appropriate evaluation.

### M09 — Agentic RAG — `TODO`

Implement planner, search strategy, tool/source router, retrieval loop, evidence evaluator, retry/stop policy, memory/state, then multi-agent variants. Evaluate task success, tool precision, steps, unnecessary actions, recovery, grounding, latency, and cost.

### M10 — Training and Production RAG — `TODO`

Study/implement retriever fine-tuning, hard-negative mining, learned rerankers, end-to-end RAG concepts, caching, incremental indexing, observability, permissions, adversarial retrieval defenses, freshness policies, scaling, and serving. Evaluate offline quality, system performance, robustness/security, freshness, and regressions.

## Immediate next step

Continue M02 with a **pretrained general-purpose semantic embedding baseline** so the tiny supervised dual encoder can be compared against a model with broad language pretraining before moving to SPLADE and ColBERT.
