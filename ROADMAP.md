# RAG Practice — Learning Roadmap

This file is the source of truth for the learning sequence in this repository. The roadmap may evolve as the field evolves, but milestone order and completion status should be updated here rather than being left only in chat, notes, or commit history.

## Goal

Learn Retrieval-Augmented Generation by implementing the mechanisms directly, measuring them, comparing alternatives on shared benchmarks, and only then introducing higher-level frameworks or production abstractions.

The repository should make it possible to answer not only **“does this RAG pipeline work?”**, but also:

- Why does it work?
- Which component improved or degraded quality?
- What does it cost in latency, tokens, memory, and index size?
- On which query classes does it fail?
- Is the improvement statistically/reproducibly meaningful on the chosen benchmark?

## Non-negotiable principles

### 1. Evaluation is mandatory

**Every milestone and every lab must include evaluation.** Evaluation is part of the implementation, not a final polishing step.

A lab is incomplete if it contains only a demo or a working pipeline without measurable comparison.

### 2. Baseline before improvement

Every new technique must be compared against the simplest relevant prior milestone. Examples:

- dense retrieval vs. BM25
- hybrid retrieval vs. dense and BM25
- reranking vs. retrieval without reranking
- HyDE vs. the same retriever with the original query
- CRAG vs. the same underlying RAG pipeline without correction
- GraphRAG vs. an appropriate flat/hierarchical retrieval baseline

### 3. Shared benchmarks for comparable experiments

Where techniques solve the same task, reuse the same corpus, question set, relevance labels, and generation-evaluation protocol. Changing both the method and the benchmark at the same time makes conclusions weak.

### 4. Learn low-level mechanisms first

For early milestones, avoid hiding the core algorithm behind LangChain, LlamaIndex, or similar orchestration frameworks. Libraries are allowed for models, tokenization, embeddings, numerical operations, and storage, but the retrieval/ranking/control logic should remain visible.

Framework reproductions can be added later as secondary implementations.

### 5. Observable pipelines

Every RAG run should be inspectable. As the repository matures, a run should expose at least:

```text
question
→ transformed query/queries
→ retrieved documents + retrieval scores
→ reranked documents + reranking scores
→ selected context
→ generated answer
→ citations / supporting evidence
→ evaluation metrics
→ latency / tokens / cost
```

### 6. Reproducibility

Experiments should record the dataset version, model names, embedding model, chunking configuration, retrieval parameters, random seed where applicable, and evaluation configuration.

## Evaluation contract

Each lab must define the following before it can be marked complete.

### A. Hypothesis

State what the technique is expected to improve and why.

Example: “Hybrid BM25 + dense retrieval should improve Recall@K on a benchmark containing both exact-name/entity queries and semantic paraphrases.”

### B. Baseline

Choose at least one earlier implementation that answers the same task.

### C. Dataset / benchmark

Document:

- corpus
- query/question set
- relevance labels or reference answers
- train/dev/test split if relevant
- any generated/synthetic data and how it was produced

### D. Retrieval metrics

Use the subset appropriate for the lab, with common metrics including:

- Hit Rate / Success@K
- Recall@K
- Precision@K
- MRR
- MAP when appropriate
- nDCG@K

Retrieval-focused milestones should not be judged only by final LLM answers.

### E. Generation / RAG metrics

Use the subset appropriate for the task, including:

- answer correctness
- faithfulness / groundedness
- context relevance
- citation precision
- citation recall / evidence coverage
- refusal or unsupported-answer rate when relevant

LLM-as-a-judge may be used, but judge prompts/models/configuration must be versioned and, where practical, checked against deterministic or human-labeled examples.

### F. System metrics

Track relevant operational trade-offs:

- end-to-end latency
- retrieval latency
- generation latency
- prompt/input tokens
- output tokens
- estimated monetary cost when paid APIs are used
- index build time
- index size / memory footprint

Not every lab must optimize these metrics, but regressions should be visible.

### G. Error analysis

Every evaluation report should include representative failure cases and classify the failure source when possible:

- bad query understanding/transformation
- retrieval miss
- ranking failure
- bad chunk/context construction
- insufficient evidence
- generation hallucination
- citation/evidence mismatch
- control-flow/routing error

### H. Result artifact

Each completed lab should leave behind a reproducible result artifact, for example:

```text
labs/<lab>/README.md
labs/<lab>/results/<experiment>.json
labs/<lab>/results/<experiment>.md
```

The report must include configuration, metrics, baseline comparison, and short findings.

## Definition of Done for every lab

A lab is **DONE** only when all applicable items below are satisfied:

- [ ] learning objective is written
- [ ] algorithm/pipeline is implemented
- [ ] core behavior has automated tests
- [ ] baseline is identified and runnable
- [ ] benchmark/evaluation dataset is defined
- [ ] retrieval evaluation is implemented when retrieval is involved
- [ ] generation/groundedness evaluation is implemented when generation is involved
- [ ] latency/resource metrics are recorded where meaningful
- [ ] experiment configuration is reproducible
- [ ] results are saved rather than shown only in terminal output
- [ ] failure cases are inspected
- [ ] findings and trade-offs are written down

## Milestone roadmap

Statuses:

- `TODO` — not started
- `IN PROGRESS` — current learning/implementation work
- `DONE` — satisfies the Definition of Done above

### M00 — Information Retrieval Fundamentals — `TODO`

Purpose: build the retrieval/evaluation foundation needed to understand later RAG systems.

Implement and learn:

- document/query representation
- tokenization basics
- inverted index
- TF / IDF / TF-IDF
- cosine similarity
- BM25 from the formula up
- dense-vector similarity basics
- top-k retrieval
- relevance judgments
- Hit Rate / Recall@K / Precision@K / MRR / MAP / nDCG@K

Evaluation gate:

- create a small deterministic IR benchmark with relevance labels
- verify metric implementations against hand-computable examples
- compare at least TF-IDF/cosine and BM25
- perform query-level error analysis

### M01 — Naive RAG From Scratch — `TODO`

Pipeline:

```text
documents → chunks → embeddings → vector index → top-k retrieval → context prompt → LLM
```

Implement:

- `Document` / `Chunk` data model
- fixed-size chunking
- embedding interface
- minimal in-memory vector index
- dense retriever
- prompt/context construction
- generator interface
- citations to retrieved chunks
- end-to-end tracing

Evaluation gate:

- retrieval metrics independent of answer generation
- answer correctness + groundedness
- citation/evidence checks
- latency/token measurements
- compare RAG against a no-retrieval generator baseline where meaningful

### M02 — Retrieval Families — `TODO`

Implement as comparable retrievers:

- BM25 / sparse lexical retrieval
- dense retrieval
- hybrid sparse + dense retrieval
- Reciprocal Rank Fusion (RRF)
- learned sparse retrieval (e.g. SPLADE) as an advanced sub-lab
- late-interaction retrieval (e.g. ColBERT/ColBERTv2) as an advanced sub-lab

Evaluation focus:

- Recall@K, MRR, nDCG@K
- lexical/entity queries vs. semantic/paraphrase queries
- latency/index-size trade-offs

### M03 — Indexing and Chunking — `TODO`

Implement and compare:

- fixed-size chunks
- overlap strategies
- sentence/paragraph-aware chunks
- semantic chunking
- metadata-enriched chunks
- parent-child retrieval
- hierarchical summaries/indexes

Evaluation focus:

- retrieval quality vs. chunk size/granularity
- evidence completeness
- context redundancy
- token budget utilization

### M04 — Reranking and Context Construction — `TODO`

Implement:

- retrieve-many / rerank-few pipeline
- cross-encoder reranking
- LLM-based reranking
- Maximum Marginal Relevance (MMR)
- duplicate/redundancy control
- context ordering and packing

Evaluation focus:

- ranking metrics before/after reranking
- answer quality at fixed context-token budgets
- latency-quality trade-off

### M05 — Query Transformation — `TODO`

Implement:

- query rewriting / Rewrite-Retrieve-Read
- multi-query retrieval
- RAG-Fusion
- Query2Doc-style expansion
- HyDE
- query decomposition

Evaluation focus:

- original query vs. transformed query retrieval metrics
- per-query-class wins/losses
- extra generation cost introduced by transformation

### M06 — Multi-hop, Active, Adaptive, and Self-correcting RAG — `TODO`

Implement progressively:

- multi-hop / iterative retrieval
- FLARE-style active retrieval concepts
- routing between no-RAG / single-shot / iterative retrieval
- Adaptive-RAG concepts
- Corrective RAG (CRAG) concepts
- Self-RAG-style retrieve/critique/reflection control

Evaluation focus:

- simple vs. multi-hop query subsets
- unnecessary-retrieval rate
- correction success rate
- hallucination/unsupported-answer rate
- cost and latency of additional control loops

### M07 — Hierarchical, Graph, and Memory-oriented RAG — `TODO`

Implement progressively:

- RAPTOR-style recursive summaries/tree retrieval
- knowledge-graph retrieval fundamentals
- GraphRAG-style entity/community indexing and local/global retrieval
- LightRAG-style graph + vector ideas
- KAG-style structured reasoning concepts
- HippoRAG-style associative graph retrieval concepts
- memory-oriented retrieval patterns

Evaluation focus:

- local fact questions vs. global corpus questions
- multi-hop relation questions
- flat-vector baseline vs. hierarchy/graph
- index construction cost and update complexity

### M08 — Specialized Sources and Modalities — `TODO`

Sub-labs:

- Web RAG
- SQL / structured-data RAG
- metadata/filter-aware RAG
- Code RAG
- multimodal RAG
- visual-document/page-image RAG
- long-context vs. retrieval routing

Evaluation focus must be modality/source appropriate while preserving the common retrieval, grounding, latency, and cost measurements where applicable.

### M09 — Agentic RAG — `TODO`

Implement:

- planner
- query/search strategy selection
- tool/source router
- retrieval loop
- evidence evaluator
- retry / stop policy
- memory/state
- multi-agent variants only after a single-agent controller is understood

Evaluation focus:

- task success
- tool/retrieval precision
- number of steps/tool calls
- unnecessary-action rate
- recovery from failed retrieval
- answer grounding
- latency and cost

### M10 — Training and Production RAG — `TODO`

Study/implement selectively:

- retriever fine-tuning
- hard-negative mining
- learned rerankers
- end-to-end / jointly trained RAG concepts
- caching
- incremental indexing
- observability
- access control / document-level permissions
- prompt-injection and adversarial-retrieval defenses
- freshness/update policies
- scaling and serving trade-offs

Evaluation focus:

- offline benchmark quality
- online/system performance
- robustness/security evaluation
- freshness and regression testing

## Repository shape — target, not all created upfront

```text
rag-practice/
├── README.md
├── ROADMAP.md
├── pyproject.toml
├── src/
│   └── rag_practice/
│       ├── core/
│       ├── retrieval/
│       ├── indexing/
│       ├── query/
│       ├── evaluation/
│       └── tracing/
├── labs/
│   ├── 00_ir_fundamentals/
│   ├── 01_naive_rag/
│   └── ...
├── datasets/
├── benchmarks/
└── tests/
```

Folders should be introduced only when the corresponding milestone needs them; do not scaffold the whole repository with empty placeholders.

## Suggested experiment record

Each experiment should eventually be serializable to a common schema similar to:

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

Start **M00 — Information Retrieval Fundamentals**. Do not move to M01 until the evaluation utilities and the small deterministic benchmark in M00 are trustworthy, because every later milestone depends on them.
