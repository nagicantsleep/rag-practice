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

### M02 — Retrieval Families — `DONE`

M02 compares retrieval families under the same held-out exact/semantic benchmark and separates representation quality from storage/index mechanics.

Implemented and evaluated:

- BM25 lexical retrieval
- hashing-vector dense-storage mechanics baseline
- supervised neural dual encoder trained from scratch
- Reciprocal Rank Fusion (RRF)
- weighted sparse+dense score fusion tuned only on a separate dev split
- transparent SPLADE-style learned-sparse mechanics
- transparent ColBERT-style MaxSim mechanics
- pretrained `sentence-transformers/all-MiniLM-L6-v2`
- full pretrained SPLADE-family `naver/splade-v3-distilbert`
- full pretrained `colbert-ir/colbertv2.0` via PyLate exhaustive MaxSim
- deterministic 10 → 100 → 1000 candidate-set scaling stress test
- representation/index footprint and CPU latency sanity measurements

Held-out checkpoint summary:

| Method | Recall@1 | Exact R@1 | Semantic R@1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.800 | 1.000 | 0.600 | 0.900 | 0.850 |
| Hashing vector | 0.800 | 1.000 | 0.600 | 0.800 | 0.835 |
| Tiny neural dual encoder | 0.800 | 0.800 | 0.800 | 1.000 | — |
| Tiny weighted hybrid | **0.900** | **1.000** | **0.800** | **1.000** | — |
| MiniLM pretrained dense | **0.900** | **1.000** | **0.800** | 0.900 | 0.925 |
| SPLADE pretrained sparse | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |
| ColBERTv2 pretrained late interaction | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |

Important findings:

- **A dense vector index does not create semantics; learned/pretrained representations do.** The hashing baseline remains lexical-like, while learned models recover vocabulary-mismatch queries.
- **Aggregate scores hide different errors.** MiniLM fixes preserved query `s1` (`conceptual likeness between paraphrases → d5`) but misses `s2`; SPLADE and ColBERTv2 rank `s1` second but solve `s2` at rank 1.
- **Pretrained contextual backbones matter.** The earlier mechanism-only SPLADE/ColBERT implementations underperform their real checkpoints; the formulas alone are not the research systems.
- **Hybrid is not automatically better.** With MiniLM, dev-set tuning selects BM25 weight `0.0`; adding sparse score hurts or adds no value on this tiny dev set. The earlier from-scratch neural model did benefit from sparse+dense fusion.
- **Representations have different serving costs.** Dense stores one vector/document; SPLADE stores sparse vocabulary activations; ColBERT stores many token vectors/document. ColBERTv2 used 211 document token vectors / 108,032 logical embedding bytes on the 10-document corpus; SPLADE averaged 183.2 non-zero values/document.
- **Candidate-set size matters.** In the deterministic stress test, MiniLM quality stays at Recall@1 `0.9` from 10 to 1000 docs while the educational Python hashing scan falls from `0.8` to `0.7` and grows from roughly `0.4 ms` to `31 ms/query`. These are implementation sanity measurements, not ANN serving claims.

Evaluation evidence:

- Final GitHub Actions PR run `32395089427` completed successfully.
- Final CI run: **32 tests passed**; pretrained dense, SPLADE checkpoint, scaling stress test, PyLate installation, and ColBERTv2 checkpoint steps all succeeded.
- Model revisions are resolved/pinned in result artifacts.
- Generation evaluation is **not applicable** to M02 because this milestone intentionally isolates retrieval quality; M01 already established separate end-to-end generation evaluation.

Artifacts: `benchmarks/m02_retrieval/`, `src/rag_practice/retrieval/`, `labs/02_retrieval_families/`, and `.github/workflows/m02-pretrained-eval.yml`.

### M03 — Indexing and Chunking — `DONE`

M03 holds BM25 scoring fixed and changes only chunk/index representation so boundary, metadata, parent expansion, and hierarchy effects remain attributable.

Implemented and evaluated:

- fixed 24-word chunks without overlap
- fixed 24-word chunks with 8-word overlap
- sentence-aware packing
- paragraph-aware packing
- deterministic sentence-boundary similarity chunking
- metadata-enriched sentence chunks
- sentence-child retrieval with paragraph-parent expansion
- document-level metadata+body routing followed by plain sentence-leaf retrieval
- evidence completeness, context redundancy/utilization, relevant-context fraction, route accuracy, searchable-index footprint, build latency, and query-latency sanity measurements

Phase-1 held-out summary:

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Source-token utilization@3 |
| --- | ---: | ---: | ---: | ---: |
| Fixed 24 | **1.000** | 0.200 | 0.800 | **1.000** |
| Fixed 24 + overlap 8 | **1.000** | 0.400 | **1.000** | 0.886 |
| Sentence 35 | 0.800 | **0.800** | **1.000** | **1.000** |
| Paragraph 80 | 0.800 | **0.800** | **1.000** | **1.000** |
| Hashing-similarity semantic boundaries | 0.800 | 0.200 | 0.800 | **1.000** |
| Sentence 35 + metadata | **1.000** | **0.800** | **1.000** | 0.635 |

Phase-2 held-out summary:

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Utilization@3 | Searchable words | Route Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 24 + overlap 8 | **1.000** | 0.400 | **1.000** | 0.886 | 367 | — |
| Sentence 35 + metadata | **1.000** | 0.800 | **1.000** | 0.635 | 434 | — |
| Parent-child | 0.800 | 0.600 | **1.000** | **1.000** | **271** | — |
| Hierarchical metadata root → plain sentence leaf | **1.000** | **1.000** | **1.000** | **1.000** | 601 | **1.000** |

Important findings:

- **Document hit is not evidence completeness.** Fixed 24-word chunks retrieve the right document for every query at rank 1 but contain all required evidence at rank 1 only 20% of the time.
- **Overlap buys coverage with context duplication.** Eight-word overlap lifts Evidence@3 from `0.8` to `1.0`, while source-token utilization falls from `1.0` to `0.886`.
- **Natural boundaries help evidence packaging on this corpus.** Sentence and paragraph packing reach Evidence@1 `0.8`, but both rank the Tropical deployment above the Arctic deployment for the metadata-dependent query because the distinguishing region is absent from body text.
- **Repeating metadata in every leaf works but spends context budget.** Metadata-enriched sentence chunks restore Doc Hit@1 `1.0` while utilization drops to `0.635`.
- **A simplistic “semantic” boundary heuristic is not automatically better.** The deterministic hashing-similarity strategy over-splits to 19 chunks and falls back to Evidence@1 `0.2`; it is retained as a negative result rather than tuned away.
- **Parent-child expansion is useful but not sufficient for routing.** It reaches Evidence@1 `0.6` with perfect utilization and the smallest searchable representation in phase 2, but still selects the Tropical parent first for the Arctic query because metadata is absent from the child search layer.
- **Metadata belongs naturally in a routing layer when answer context should stay clean.** Hierarchical metadata+body roots route every held-out query correctly, then return plain sentence leaves with Doc Hit@1/Evidence@1/utilization all `1.0`. The trade-off is a larger searchable representation: 601 words versus 367 for the overlap baseline and 434 for metadata-enriched flat chunks.
- **These results are controlled-mechanism evidence, not a universal chunking leaderboard.** The benchmark is intentionally tiny and timings are GitHub Actions CPU sanity measurements.

Evaluation evidence:

- Final phase-2 GitHub Actions PR run `32407289218` completed successfully.
- Final CI run: **39 tests passed**; phase-1 and phase-2 evaluation steps both succeeded.
- Results are persisted as JSON and Markdown; representative Arctic/Tropical ambiguity and semantic over-splitting failures are retained in the reports.
- Generation evaluation is **not applicable** to M03 because this milestone isolates retrieval/index representation and context-selection quality.

Artifacts: `benchmarks/m03_chunking/`, `src/rag_practice/indexing/`, `src/rag_practice/evaluation/chunking.py`, `labs/03_indexing_chunking/`, and `.github/workflows/m03-indexing-chunking.yml`.

### M04 — Reranking and Context Construction — `TODO`

Implement retrieve-many/rerank-few, cross-encoder reranking, LLM reranking, MMR, redundancy control, and context ordering/packing. Evaluate ranking changes, answer quality at fixed context budgets, and latency-quality trade-offs.

### M05 — Query Transformation — `TODO`

Implement query rewriting, multi-query retrieval, RAG-Fusion, Query2Doc-style expansion, HyDE, and query decomposition. Evaluate original vs transformed retrieval, per-query-class wins/losses, and transformation cost.

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

Start **M04 — Reranking and Context Construction**. Freeze the first-stage retrieval candidate set where possible, then vary reranking and context-selection/packing policies so ranking gains, redundancy reduction, answer quality, latency, and context-budget effects can be attributed independently.
