# M04 Completion Summary

Status: **DONE**  
Benchmark control: frozen BM25 candidate sets over metadata-enriched sentence chunks  
Final evaluation CI before completion docs: `32410053158` — **51 tests passed**; phase 1, phase 2, and candidate-depth sweep all succeeded.

## What M04 isolates

M04 separates first-stage recall from reranking and context construction. Every learned reranker only reorders candidates already returned by BM25, and candidate document/evidence recall is recorded before reranking. Context packing and ordering are then evaluated separately so ranking gains are not confused with token-budget or ordering effects.

## Phase 1 — reranking and context selection

Frozen BM25 top-6 candidate document/evidence recall were both `1.0`.

| Method | Evidence@1 | Evidence@3 | Source util@3 | Relevant context@3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 first-stage | 0.800 | 1.000 | 0.635 | 0.742 |
| Cross-encoder | 0.800 | 1.000 | 0.632 | 0.814 |
| Cross-encoder + MMR | 0.800 | 1.000 | 0.639 | 0.799 |
| Cross-encoder + 100-word packing | 0.800 | 1.000 | 0.654 | **0.904** |

The cross-encoder improves context purity without changing the already-saturated evidence recall. MMR at the chosen fixed setting is approximately neutral on this tiny corpus. Budget packing produces the densest relevant context.

## Phase 2 — LLM reranking, ordering, and generation

Models are pinned:

- `cross-encoder/ms-marco-MiniLM-L6-v2` @ `c5f2b386de279a97c53a702dd5189d1c407160dc`
- `google/flan-t5-small` @ resolved commit `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`

| Policy | Relevant context@3 | Mean context words | Extractive F1 | FLAN F1 | FLAN grounded-token recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| first-stage top-3 | 0.742 | 112.2 | 0.331 | 0.382 | 0.775 |
| cross-encoder top-3 | 0.814 | 111.6 | **0.387** | **0.700** | 0.775 |
| cross + pack 100, relevance order | **0.904** | 84.4 | **0.387** | 0.452 | 0.575 |
| cross + pack 100, source order | **0.904** | 84.4 | 0.341 | 0.452 | 0.575 |
| cross + pack 100, edge order | **0.904** | 84.4 | **0.387** | 0.452 | 0.575 |
| pointwise FLAN rerank + pack 100 | 0.819 | 83.2 | **0.387** | 0.452 | 0.575 |

Cross-encoder top-3 is the best FLAN generation policy on this benchmark. Packing raises relevant-context density and cuts context length, but it does **not** improve the small instruction generator: aggregate FLAN F1 falls from `0.700` to `0.452`. This negative result is retained instead of tuning the test-set budget.

Pointwise FLAN reranking is also not competitive here: mean CPU reranking latency is roughly `323 ms/query`, versus roughly `68 ms/query` for the cross-encoder, while its packed relevant-context fraction is lower (`0.819` vs `0.904`).

## Candidate-depth latency/quality sweep

| Candidate k | Candidate evidence recall | Evidence@3 after rerank | Relevant context@3 | Approx. mean rerank ms |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.000 | 1.000 | **0.833** | ~34–38 |
| 4 | 1.000 | 1.000 | 0.758 | ~48 |
| 6 | 1.000 | 1.000 | 0.814 | ~69–71 |

This benchmark is already saturated at candidate `k=2`. Retrieving more candidates does not improve candidate recall or Evidence@3 and roughly doubles reranking latency by `k=6`. The lesson is to tune retrieve-many/rerank-few depth against measured recall, not assume a larger candidate set is automatically better.

## Representative failures and caveats

- For the latency-budget query, BM25 top-3 causes FLAN to answer the wrong idea (`to improve ordering`); cross-encoder reranking moves the actual latency evidence forward and FLAN reaches F1 `1.0`. This is a direct ranking-to-generation win.
- For the overlapping-chunk question, cross-encoder top-3 gives FLAN an exact answer, while the packed two-chunk context causes FLAN-T5-small to emit `[2]`. The selected evidence remains sufficient, but the generator/prompt fails. Context quality and generator robustness must therefore be measured separately.
- For the natural-boundaries question, FLAN emits bracket labels such as `[1]`/`[2]` despite relevant evidence being present. This is retained as a generator/prompt failure rather than hidden by retrieval metrics.
- The deterministic extractive answerer is fully grounded by construction and shows citation precision improving from `0.9` to `1.0` after cross-encoder reranking, but its lexical sentence-selection heuristic is intentionally simple and is not an LLM-quality benchmark.
- `grounded_token_recall` is an exact-token metric; a semantically supported paraphrase can score below `1.0`. It is a reproducible signal, not a semantic faithfulness oracle.
- Source-order and edge-order experiments operate on the exact same packed candidate set. No universal ordering advantage appears on this five-query corpus; source order can lower rank-1 evidence/extractive answer quality even though Evidence@3 is unchanged.
- All timings are GitHub Actions CPU sanity measurements, not production throughput claims.

## Evaluation contract

- hypothesis and explicit baselines: present
- benchmark and frozen-candidate control: present
- retrieval/ranking/context metrics: present
- generation correctness, grounding, and citation metrics: present
- latency-quality trade-off: present
- model revisions and deterministic settings: persisted
- machine-readable JSON and human-readable Markdown: persisted
- representative failures and negative results: retained
- automated regression suite: **51 tests passed** in CI run `32410053158`

Artifacts: `benchmarks/m04_context/`, `src/rag_practice/reranking/`, `src/rag_practice/models/flan_t5.py`, `src/rag_practice/generation/query_extract.py`, `labs/04_reranking_context/`, and `.github/workflows/m04-reranking-context.yml`.
