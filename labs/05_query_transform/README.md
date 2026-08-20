# Lab 05 — Query Transformation

M05 freezes the corpus and retriever family within each comparison, then changes only the query-side representation/control. The goal is to measure when rewriting/expansion helps vocabulary mismatch or multi-aspect needs and when it causes intent drift.

## Benchmark

Corpus: `benchmarks/m00_ir/corpus.jsonl`  
Queries: `benchmarks/m05_query_transform/queries.jsonl`

The 12 held-out queries are explicitly classified as:

- `exact` — strong lexical match should already work
- `semantic` — vocabulary/paraphrase mismatch
- `underspecified` — intent is recognizable but expressed indirectly
- `multi_aspect` — two relevant documents must be recovered

For multi-aspect queries, `complete_recall@3` is reported in addition to ordinary Recall@K. It is 1 only if **every** relevant document appears in top 3, so partial retrieval cannot masquerade as success.

## Methods and fair baselines

BM25 is fixed for:

- original query baseline
- single generative rewrite
- multi-query score fusion
- RAG-Fusion using reciprocal-rank fusion
- Query2Doc-style pseudo-document expansion
- decomposition + RRF

MiniLM dense retrieval is fixed for:

- original dense query baseline
- HyDE hypothetical-document query representation

HyDE is therefore compared with `dense_original`, **not** with BM25. All BM25 query transformations are compared with `bm25_original`.

## Transformation model

Pinned `google/flan-t5-small` is used only to generate transformed search representations. Qrels/relevant document IDs are never included in transformation prompts. Generated outputs are persisted per query so drift and bad transformations can be inspected rather than hidden by aggregate metrics.

Multi-query always retains the original query as one member. Query2Doc retains the original query plus the generated pseudo-document. Empty rewrite/HyDE/decomposition outputs fall back to the original query.

## Metrics

- MRR / MAP
- Recall@1/@3/@5
- nDCG@1/@3/@5
- complete Recall@3
- class-specific exact/semantic/underspecified/multi-aspect metrics
- transformation latency
- end-to-end retrieval latency
- generated-word count as a deterministic transformation-cost proxy
- representative per-query wins and regressions

Generation answer-quality evaluation is not applicable to M05: generation is being studied as a **query transformation mechanism**, while final answer generation is held out so retrieval effects remain visible. Transformation output/cost and downstream retrieval quality are evaluated directly.

## Completion gate

M05 is not `DONE` until:

- the full regression suite passes;
- all six transformation families plus their fair baselines run in CI;
- JSON + Markdown results are persisted;
- query-class wins/regressions and transformation outputs are inspected;
- latency/output-cost trade-offs are reported;
- `ROADMAP.md` records final findings and CI evidence.
