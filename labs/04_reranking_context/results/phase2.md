# M04 Phase 2 — LLM Reranking, Context Ordering, and Answer Quality

Instruction model: `google/flan-t5-small` @ `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` (requested `0fc9ddf`).

All policies share the same frozen BM25 top-6 candidate sets. Source-order and edge-order reuse the exact selected set from cross-encoder budget packing; only ordering changes.

| Policy | Evidence@1 | Evidence@3 | Relevant ctx@3 | Ctx words | Extractive F1 | Extractive grounded | FLAN F1 | FLAN grounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first_stage_top3 | 0.800 | 1.000 | 0.742 | 112.2 | 0.331 | 1.000 | 0.382 | 0.775 |
| cross_encoder_top3 | 0.800 | 1.000 | 0.814 | 111.6 | 0.387 | 1.000 | 0.700 | 0.775 |
| cross_pack100_relevance | 0.800 | 1.000 | 0.904 | 84.4 | 0.387 | 1.000 | 0.452 | 0.575 |
| cross_pack100_source_order | 0.600 | 1.000 | 0.904 | 84.4 | 0.341 | 1.000 | 0.452 | 0.575 |
| cross_pack100_edge_order | 0.800 | 1.000 | 0.904 | 84.4 | 0.387 | 1.000 | 0.452 | 0.575 |
| llm_pack100_relevance | 0.800 | 1.000 | 0.819 | 83.2 | 0.387 | 1.000 | 0.452 | 0.575 |

Mean cross-encoder rerank latency: **67.35 ms/query**  
Mean pointwise FLAN rerank latency: **325.13 ms/query**

Extractive generation is qrel-blind and deterministic. FLAN receives only question + ordered context. References are used only after generation for evaluation.
