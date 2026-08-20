# M05 Query Transformation — baseline

Transformer: `google/flan-t5-small` @ `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`.

BM25-family methods are compared with `bm25_original`; HyDE is compared with `dense_original` because HyDE changes the query representation but keeps the dense retriever/index fixed.

| Method | R@1 | R@3 | Complete R@3 | Exact R@1 | Semantic R@1 | Underspecified R@1 | Multi-aspect R@3 | Mean total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_original | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 0.06 |
| bm25_rewrite | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 284.83 |
| multi_query_score_fusion | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 239.84 |
| rag_fusion_rrf | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 239.85 |
| query2doc_bm25 | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 1104.16 |
| decomposition_rrf | 0.167 | 0.250 | 0.167 | 0.333 | 0.000 | 0.333 | 0.333 | 505.14 |
| dense_original | 0.792 | 0.917 | 0.917 | 1.000 | 0.667 | 1.000 | 1.000 | 10.94 |
| hyde_dense | 0.625 | 0.833 | 0.750 | 1.000 | 0.333 | 0.667 | 0.667 | 431.20 |

`complete_recall@3` requires every relevant document to be present, which matters for multi-aspect queries where ordinary hit-rate can hide partial retrieval.
