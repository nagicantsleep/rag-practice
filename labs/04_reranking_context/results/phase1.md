# M04 Phase 1 — Cross-encoder, MMR, and Context Packing

Model: `cross-encoder/ms-marco-MiniLM-L6-v2` @ `c5f2b386de279a97c53a702dd5189d1c407160dc`

Frozen BM25 candidate set: top-6; returned context: top-3; packing budget: 100 words.

Candidate document recall@6: **1.000**  
Candidate evidence recall@6: **1.000**

| Method | Doc hit@1 | Evidence@1 | Evidence@3 | Source util@3 | Relevant context@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| first_stage | 1.000 | 0.800 | 1.000 | 0.635 | 0.742 |
| cross_encoder | 1.000 | 0.800 | 1.000 | 0.632 | 0.814 |
| cross_encoder_mmr | 1.000 | 0.800 | 1.000 | 0.639 | 0.799 |
| cross_encoder_budget_pack | 1.000 | 0.800 | 1.000 | 0.654 | 0.904 |

The candidate-recall rows are a guardrail: reranking cannot repair evidence that the first stage failed to retrieve. MMR and packing are evaluated separately from the cross-encoder ordering so diversity and budget effects stay visible.
