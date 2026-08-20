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

## Transformation models and capacity control

The primary mechanism run uses pinned `google/flan-t5-small` only to generate transformed search representations. Qrels/relevant document IDs are never included in transformation prompts. Generated outputs are persisted per query so drift and bad transformations can be inspected rather than hidden by aggregate metrics.

Because the small model produced severe decomposition/HyDE failures, M05 adds a controlled capacity experiment with pinned `google/flan-t5-base`. The prompts, benchmark, retrievers, index, evaluation, and fallback rules are unchanged; **only transformer capacity changes**. No held-out prompt tuning is performed after seeing results.

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

## Results

### FLAN-T5-small

| Method | R@1 | R@3 | Complete R@3 |
| --- | ---: | ---: | ---: |
| BM25 original | 0.792 | 0.875 | 0.833 |
| Rewrite | 0.792 | 0.875 | 0.833 |
| Multi-query score fusion | 0.792 | 0.875 | 0.833 |
| RAG-Fusion / RRF | 0.792 | 0.875 | 0.833 |
| Query2Doc + BM25 | 0.792 | 0.875 | 0.833 |
| Decomposition + RRF | 0.167 | 0.250 | 0.167 |
| Dense original | 0.792 | 0.917 | 0.917 |
| HyDE + dense | 0.625 | 0.833 | 0.750 |

### FLAN-T5-base capacity control

| Method | R@1 | R@3 | Complete R@3 |
| --- | ---: | ---: | ---: |
| BM25 original | 0.792 | 0.875 | 0.833 |
| Rewrite | 0.792 | 0.875 | 0.833 |
| Multi-query score fusion | 0.792 | 0.875 | 0.833 |
| RAG-Fusion / RRF | 0.792 | 0.875 | 0.833 |
| Query2Doc + BM25 | 0.792 | 0.833 | 0.750 |
| Decomposition + RRF | 0.625 | 0.667 | 0.667 |
| Dense original | 0.792 | 0.917 | 0.917 |
| HyDE + dense | 0.708 | 0.917 | 0.917 |

## Findings

- **No generative transformation beats its fair original-query baseline on aggregate.** Rewrite, multi-query, and RAG-Fusion are quality-neutral here while adding generation latency.
- **Capacity helps, but does not solve reliability.** FLAN-T5-base improves decomposition and HyDE substantially relative to FLAN-T5-small, yet both remain below their original-query baselines at rank 1 and pathological outputs remain.
- **Query2Doc can regress multi-aspect completeness.** The base capacity run drops overall R@3/complete Recall@3 and multi-aspect R@3 relative to BM25 original.
- **The preserved vocabulary-mismatch paraphrase is still best solved by pretrained dense retrieval.** Surface rewriting does not bridge that gap on this benchmark.
- **Decomposition must be conditional.** Small/simple queries do not benefit, and malformed subquestions can destroy retrieval; this directly motivates adaptive routing in M06.
- **HyDE can cause semantic drift.** A fluent hypothetical document is not necessarily a useful retrieval representation.
- **Transformation cost must be evaluated with quality.** BM25/dense original-query retrieval is far cheaper than the generative paths in these CPU runs, so a transformation should earn its cost through measurable gains.
- **Negative results are retained rather than tuned away.** See `results/m05_summary.md` and the per-query JSON traces for representative failures.

## Evaluation evidence

- GitHub Actions capacity-control mechanism run `32413220357`: success.
- GitHub Actions completed documentation/ROADMAP run `32414001399`: success.
- **59 tests passed** on the completed tree.
- FLAN-T5-small baseline and FLAN-T5-base capacity-control evaluations both succeeded.
- JSON + Markdown results persist model revisions, transformed queries, rankings, per-class metrics, latency, and generated-word counts.

## Completion gate

- [x] full regression suite passes
- [x] all six transformation families plus fair baselines run in CI
- [x] JSON + Markdown results are persisted
- [x] query-class wins/regressions and transformation outputs are inspected
- [x] latency/output-cost trade-offs are reported
- [x] transformer-capacity confound is tested without held-out prompt tuning
- [x] representative failures are retained
- [x] completion summary is written
- [x] completed documentation/ROADMAP head passes CI

M05 is `DONE`. The remaining merge gate is a final CI pass on this checklist-closing head; no further mechanism or documentation changes are planned.
