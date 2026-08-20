# RAG Practice — Learning Roadmap

This file is the source of truth for the learning sequence in this repository. Update milestone status here so the plan does not live only in chat or commit history.

## Goal

Learn Retrieval-Augmented Generation by implementing mechanisms directly, evaluating them on shared benchmarks, comparing them against explicit baselines, and only then introducing higher-level frameworks or production abstractions.

The repository should answer more than “does it run?”:

- why does a method work or fail?
- which component changed retrieval or answer quality?
- what are the latency, token, memory, index-size, and cost trade-offs?
- which query classes improve or regress?
- are conclusions reproducible on a fixed benchmark?

## Non-negotiable principles

### 1. Evaluation is mandatory

**Every milestone and every lab must include evaluation.** Evaluation is part of the implementation, not a final polishing step.

A demo or working pipeline without a baseline, benchmark, metrics, saved results, and error analysis is not complete.

### 2. Baseline before improvement

Every new technique must be compared with the simplest relevant prior implementation. Examples:

- dense retrieval vs. BM25
- hybrid vs. BM25 and dense
- reranking vs. the same retriever without reranking
- HyDE vs. the same retriever with the original query
- CRAG vs. the same RAG pipeline without correction
- GraphRAG vs. an appropriate flat or hierarchical retrieval baseline

### 3. Shared benchmarks for comparable experiments

When methods solve the same task, reuse the corpus, questions, relevance labels, and evaluation protocol. Do not change both the method and benchmark and then attribute the difference to the method.

### 4. Learn mechanisms before frameworks

For early milestones, do not hide the important algorithm behind LangChain, LlamaIndex, or another orchestration framework. Libraries are fine for models, tokenization, embeddings, numerical operations, and storage, but retrieval/ranking/control logic should remain visible.

Framework reproductions can be secondary implementations later.

### 5. Observable pipelines

As the repository grows, a run should expose at least:

```text
question
→ transformed query/queries
→ retrieved documents + scores
→ reranked documents + scores
→ selected context
→ generated answer
→ citations/evidence
→ evaluation metrics
→ latency / tokens / cost
```

### 6. Reproducibility

Record dataset version, model names/versions, embedding model, chunking parameters, retrieval parameters, random seed where applicable, and evaluator configuration.

## Evaluation contract

Each lab must define these before it can be marked `DONE`.

### A. Hypothesis

State what should improve and why.

### B. Baseline

Identify at least one earlier implementation that solves the same task.

### C. Dataset / benchmark

Document the corpus, queries/questions, relevance labels or reference answers, split if relevant, and provenance of any synthetic data.

### D. Retrieval metrics

Use the subset appropriate for the task:

- Hit Rate / Success@K
- Recall@K
- Precision@K
- MRR
- MAP where appropriate
- nDCG@K

Retrieval-focused work must not be judged only by the final LLM answer.

### E. Generation / RAG metrics

Use the subset appropriate for the task:

- answer correctness
- faithfulness / groundedness
- context relevance
- citation precision
- citation recall / evidence coverage
- unsupported-answer/refusal behavior where relevant

LLM-as-a-judge is allowed, but judge prompts/models/config must be versioned and checked against deterministic or human-labeled examples where practical.

### F. System metrics

Track relevant operational trade-offs:

- end-to-end latency
- retrieval latency
- generation latency
- input/output tokens
- estimated API cost when paid APIs are used
- index build time
- index size / memory footprint

### G. Error analysis

Classify representative failures where possible:

- query understanding/transformation
- retrieval miss
- ranking failure
- chunk/context construction
- insufficient evidence
- generation hallucination
- citation/evidence mismatch
- routing/control-flow error

### H. Saved result artifact

A completed lab should leave reproducible results, for example:

```text
labs/<lab>/README.md
labs/<lab>/results/<experiment>.json
labs/<lab>/results/<experiment>.md
```

The report must include configuration, metrics, baseline comparison, failures, and findings.

## Definition of Done for every lab

- [ ] learning objective is written
- [ ] algorithm/pipeline is implemented
- [ ] core behavior has automated tests
- [ ] baseline is identified and runnable
- [ ] benchmark/evaluation dataset is defined
- [ ] retrieval evaluation exists when retrieval is involved
- [ ] generation/groundedness evaluation exists when generation is involved
- [ ] latency/resource metrics are recorded where meaningful
- [ ] experiment configuration is reproducible
- [ ] results are saved rather than only printed
- [ ] representative failures are inspected
- [ ] findings and trade-offs are written down

## Milestone roadmap

Statuses: `TODO`, `IN PROGRESS`, `DONE`.

### M00 — Information Retrieval Fundamentals — `DONE`

Purpose: establish retrieval math and trustworthy evaluation utilities reused by every later milestone.

Implemented:

- tokenization basics
- inverted index
- TF / IDF / TF-IDF
- dense and sparse cosine similarity
- BM25 from the formula
- top-k retrieval
- relevance judgments
- Hit Rate@K / Precision@K / Recall@K / MRR / MAP / nDCG@K
- deterministic benchmark
- hand-computable metric tests
- TF-IDF vs. BM25 baseline comparison
- query-level error analysis

Artifacts:

- `src/rag_practice/ir/`
- `src/rag_practice/evaluation/`
- `benchmarks/m00_ir/`
- `labs/00_ir_fundamentals/`

Key finding: both lexical baselines intentionally fail a paraphrase query with vocabulary mismatch. That known retrieval miss becomes a target for dense retrieval.

### M01 — Naive RAG From Scratch — `TODO`

Pipeline:

```text
documents → chunks → embeddings → vector index → top-k retrieval → context prompt → LLM
```

Implement:

- `Document` / `Chunk` model
- fixed-size chunking
- embedding interface
- minimal in-memory dense vector index
- dense retriever
- prompt/context construction
- generator interface
- citations to retrieved chunks
- end-to-end tracing

Evaluation gate:

- reuse/extend the M00 benchmark
- retrieval metrics independent from answer generation
- answer correctness + groundedness
- citation/evidence checks
- latency/token measurements
- RAG vs. no-retrieval baseline where meaningful
- verify whether dense retrieval fixes the M00 paraphrase failure

### M02 — Retrieval Families — `TODO`

Implement and compare:

- BM25 / sparse lexical
- dense retrieval
- hybrid sparse + dense
- Reciprocal Rank Fusion (RRF)
- learned sparse retrieval such as SPLADE as an advanced sub-lab
- late-interaction retrieval such as ColBERT/ColBERTv2 as an advanced sub-lab

Evaluation: Recall@K, MRR, nDCG@K, query-class breakdown, latency, index size.

### M03 — Indexing and Chunking — `TODO`

Compare:

- fixed chunks
- overlap strategies
- sentence/paragraph-aware chunks
- semantic chunking
- metadata-enriched chunks
- parent-child retrieval
- hierarchical summaries/indexes

Evaluation: retrieval quality vs. granularity, evidence completeness, redundancy, token-budget use.

### M04 — Reranking and Context Construction — `TODO`

Implement:

- retrieve-many / rerank-few
- cross-encoder reranking
- LLM reranking
- Maximum Marginal Relevance (MMR)
- duplicate/redundancy control
- context ordering and packing

Evaluation: ranking before/after reranking, answer quality at fixed context budgets, latency-quality trade-off.

### M05 — Query Transformation — `TODO`

Implement:

- query rewriting / Rewrite-Retrieve-Read
- multi-query retrieval
- RAG-Fusion
- Query2Doc-style expansion
- HyDE
- query decomposition

Evaluation: original vs. transformed-query retrieval, query-class wins/losses, extra generation cost.

### M06 — Multi-hop, Active, Adaptive, and Self-correcting RAG — `TODO`

Implement progressively:

- multi-hop / iterative retrieval
- FLARE-style active retrieval concepts
- no-RAG / single-shot / iterative routing
- Adaptive-RAG concepts
- Corrective RAG (CRAG) concepts
- Self-RAG-style retrieve/critique/reflection control

Evaluation: simple vs. multi-hop subsets, unnecessary retrieval rate, correction success, unsupported-answer rate, control-loop cost/latency.

### M07 — Hierarchical, Graph, and Memory-oriented RAG — `TODO`

Implement progressively:

- RAPTOR-style recursive summaries/tree retrieval
- knowledge-graph retrieval fundamentals
- GraphRAG-style entity/community indexing and local/global retrieval
- LightRAG-style graph + vector ideas
- KAG-style structured reasoning concepts
- HippoRAG-style associative graph retrieval concepts
- memory-oriented retrieval patterns

Evaluation: local vs. global questions, multi-hop relations, flat-vector baseline vs. hierarchy/graph, construction/update cost.

### M08 — Specialized Sources and Modalities — `TODO`

Sub-labs:

- Web RAG
- SQL / structured-data RAG
- metadata/filter-aware RAG
- Code RAG
- multimodal RAG
- visual-document/page-image RAG
- long-context vs. retrieval routing

Evaluation must be source/modality appropriate while retaining common retrieval, grounding, latency, and cost measurements where applicable.

### M09 — Agentic RAG — `TODO`

Implement:

- planner
- query/search strategy selection
- tool/source router
- retrieval loop
- evidence evaluator
- retry/stop policy
- memory/state
- multi-agent variants only after the single-agent controller is understood

Evaluation: task success, tool/retrieval precision, number of steps/tool calls, unnecessary-action rate, recovery, grounding, latency, cost.

### M10 — Training and Production RAG — `TODO`

Study/implement selectively:

- retriever fine-tuning
- hard-negative mining
- learned rerankers
- end-to-end / jointly trained RAG concepts
- caching
- incremental indexing
- observability
- document-level permissions
- prompt-injection / adversarial-retrieval defenses
- freshness/update policies
- scaling and serving trade-offs

Evaluation: offline quality, online/system performance, robustness/security, freshness, regression testing.

## Target repository shape

Create folders only when their milestone needs them; avoid empty scaffolding.

```text
rag-practice/
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── src/rag_practice/
│   ├── core/
│   ├── ir/
│   ├── retrieval/
│   ├── indexing/
│   ├── query/
│   ├── evaluation/
│   └── tracing/
├── labs/
│   ├── 00_ir_fundamentals/
│   ├── 01_naive_rag/
│   └── ...
├── benchmarks/
└── tests/
```

## Suggested experiment record

```yaml
experiment_id: m02_hybrid_rrf_v1
milestone: M02
dataset: <name/version>
method:
  retriever: hybrid
  top_k: 10
  parameters: {}
models:
  embedding: <model/version>
  generator: <model/version>
metrics:
  retrieval:
    recall_at_5: 0.0
    mrr: 0.0
    ndcg_at_10: 0.0
  generation:
    correctness: 0.0
    groundedness: 0.0
  system:
    latency_ms: 0.0
    input_tokens: 0
    output_tokens: 0
baseline: <experiment_id>
notes: <short findings>
```

## Immediate next step

Start **M01 — Naive RAG From Scratch**. Keep the M00 lexical baselines and benchmark as regression references, and add generation/groundedness evaluation from the first end-to-end RAG implementation.
