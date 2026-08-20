# M05 Query Transformation — capacity_base

Transformer: `google/flan-t5-base` @ `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`.

BM25-family methods are compared with `bm25_original`; HyDE is compared with `dense_original` because HyDE changes the query representation but keeps the dense retriever/index fixed.

| Method | R@1 | R@3 | Complete R@3 | Exact R@1 | Semantic R@1 | Underspecified R@1 | Multi-aspect R@3 | Mean total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_original | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 0.05 |
| bm25_rewrite | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 733.69 |
| multi_query_score_fusion | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 655.56 |
| rag_fusion_rrf | 0.792 | 0.875 | 0.833 | 1.000 | 0.667 | 1.000 | 0.833 | 655.57 |
| query2doc_bm25 | 0.792 | 0.833 | 0.750 | 1.000 | 0.667 | 1.000 | 0.667 | 612.52 |
| decomposition_rrf | 0.625 | 0.667 | 0.667 | 1.000 | 0.333 | 1.000 | 0.333 | 1698.54 |
| dense_original | 0.792 | 0.917 | 0.917 | 1.000 | 0.667 | 1.000 | 1.000 | 9.23 |
| hyde_dense | 0.708 | 0.917 | 0.917 | 1.000 | 0.333 | 1.000 | 1.000 | 1539.69 |

`complete_recall@3` requires every relevant document to be present, which matters for multi-aspect queries where ordinary hit-rate can hide partial retrieval.
