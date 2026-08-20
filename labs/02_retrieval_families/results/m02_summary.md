# M02 Retrieval Families — Completion Summary

Status: **DONE**  
Benchmark: `benchmarks/m02_retrieval@v1`  
Final CI evidence: workflow run `32395089427`, **32 tests passed**, all checkpoint/scaling evaluation steps succeeded.

## Final held-out comparison

| Method | Recall@1 | Exact R@1 | Semantic R@1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.800 | 1.000 | 0.600 | 0.900 | 0.850 |
| Hashing vector | 0.800 | 1.000 | 0.600 | 0.800 | 0.835 |
| MiniLM pretrained dense | **0.900** | **1.000** | **0.800** | 0.900 | 0.925 |
| SPLADE pretrained sparse | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |
| ColBERTv2 late interaction | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |

The table is not sufficient to pick a winner: there are only ten held-out queries and models with nearly identical aggregate scores make different errors.

## Preserved semantic failures

`M00/M01` deliberately retained a hard paraphrase query:

```text
s1: conceptual likeness between paraphrases
relevant: d5 — dense retrieval / semantic matching
```

- BM25 and hashing miss it.
- MiniLM retrieves `d5` at rank 1.
- SPLADE and ColBERTv2 rank `d8` first but recover `d5` at rank 2.

A second semantic query exposes the opposite pattern:

```text
s2: evidence lookup combined with a text generator
relevant: d6 — retrieval augmented generation
```

- MiniLM ranks `d10` first and does not recover `d6` in top-3.
- SPLADE and ColBERTv2 rank `d6` first.

This is the strongest M02 lesson: **retriever families have complementary inductive biases and failure modes; aggregate metrics alone are not enough.**

## Representation trade-offs

- Dense MiniLM: one 384-d float32 vector/document; 15,360 logical bytes for 10 documents.
- SPLADE: interpretable sparse vocabulary activation; 1,832 non-zero values total / 183.2 per document in this run.
- ColBERTv2: 211 stored document token vectors / 108,032 logical embedding bytes on the 10-document corpus.

The system numbers explain why retrieval architecture and serving architecture are different questions. M02 studies representation/scoring; ANN, PLAID, compression, sharding, and production serving are deferred to M10.

## Scaling evidence

With deterministic off-topic distractors, candidate count grows 10 → 100 → 1000 while the target questions remain unchanged.

- MiniLM Recall@1 stays `0.9` at all three scales.
- Hashing Recall@1 drops `0.8 → 0.7 → 0.7`.
- The educational Python hashing exhaustive scan grows from roughly `0.4 ms` to `31 ms/query`.
- MiniLM's NumPy matrix scoring remains around `10–11 ms/query` at 1000 documents on the CI runner.

This is a controlled robustness/system stress test, **not** a broad retrieval benchmark or ANN performance claim.

## Hybrid negative result

Weighted BM25 + MiniLM fusion was tuned only on the separate dev set. The selected BM25 weight was **0.0**. On this small dev set, sparse score fusion did not improve the pretrained dense model. This is retained as a negative result rather than forcing a hybrid win.

## Reproducibility

Pinned/resolved model revisions:

- MiniLM: `1c82ace116a2629de82404c4be48c0e5d4cf08be`
- SPLADE v3 DistilBERT: `2db06b86d65e316e2ca9907aa1aa8be6f8c4e739`
- ColBERTv2: `c1e84128e85ef755c096a95bdb06b47793b13acf`

PyLate 1.6.0 pins Sentence Transformers 5.3.0, so the CI intentionally runs MiniLM/SPLADE under Sentence Transformers 5.6.1 first, then installs PyLate and runs ColBERT in the resulting dependency set. The dependency transition itself is exercised by CI.

## Definition of Done

- learning objective: satisfied
- implementations: lexical, dense, hybrid, learned sparse, late interaction
- automated tests: **32 passed**
- baselines: BM25 + hashing + from-scratch dense
- shared benchmark + dev/test separation: present
- retrieval metrics: saved
- generation metrics: not applicable; M02 intentionally isolates retrieval
- system/footprint metrics: saved
- reproducible model revisions: saved
- error analysis: saved
- machine-readable + human-readable result artifacts: saved

M02 is complete. The next experiment changes **index/chunk construction while holding retrieval models stable** so causal attribution remains possible.
