# Lab 04 — Reranking and Context Construction

M04 freezes the first-stage candidate set before changing reranking or context construction. This prevents improvements in first-stage recall from being incorrectly credited to the reranker.

## Phase 1 — cross-encoder and selection controls

Baseline: BM25 top-6 over metadata-enriched sentence chunks, returning top-3 context chunks.

Compared:

- first-stage BM25 ordering
- pretrained cross-encoder reranking
- cross-encoder + MMR
- cross-encoder + source-overlap-aware fixed-budget packing

Guardrail metrics include candidate document/evidence recall before reranking, Evidence@1/@3, source-token utilization, relevant-context fraction, and CPU latency.

## Phase 2 — instruction reranking, ordering, and answer quality

Uses the same frozen BM25 candidates and compares:

- cross-encoder ordering
- pointwise FLAN-T5 yes/no relevance reranking
- relevance-ordered budget packing
- source-ordering of the exact same selected set
- edge-biased ordering of the exact same selected set

Answer quality is evaluated with two generators that never receive qrels/references:

- a deterministic query-aware extractive answerer
- pinned `google/flan-t5-small` instruction generation

Post-generation metrics include token F1 against references, grounded-token recall, and extractive citation precision/recall. References are loaded only for evaluation after generation.

## Candidate-depth sweep

BM25 candidate depths `k = 2, 4, 6` are reranked with the same cross-encoder. The sweep reports candidate evidence recall, Evidence@3 after reranking, relevant-context fraction, and reranking latency to expose the retrieve-many/rerank-few trade-off.

## Reproducibility

- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L6-v2` @ `c5f2b386de279a97c53a702dd5189d1c407160dc`
- Instruction model: `google/flan-t5-small` requested at `0fc9ddf`; the resolved commit is persisted by the phase-2 runner.
- CPU inference, deterministic decoding (`do_sample=False`, one beam).
- Shared source corpus and retrieval questions: `benchmarks/m03_chunking/`.
- Answer references: `benchmarks/m04_context/questions.jsonl`.

## Completion gate

M04 is not `DONE` until the full test suite, phase 1, phase 2, and candidate-depth sweep all succeed in CI; JSON and Markdown results must be persisted; representative wins/losses and latency-quality trade-offs must be written into `ROADMAP.md` and the completion summary.
