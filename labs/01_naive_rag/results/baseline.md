# M01 Baseline Results — Naive RAG From Scratch

Experiment: `m01_naive_rag_v1`  
Benchmark: `benchmarks/m01_rag@v1`  
Corpus: reused from `benchmarks/m00_ir` (10 documents)  
Questions: 6  
Top-k: 3

## Hypothesis

A minimal RAG pipeline should improve answerability over the same context-only generator with no retrieval while keeping answers grounded in retrieved evidence. The deterministic hashing embedder is expected to work for lexical/phrase-overlap queries but **not** to solve true semantic paraphrase mismatch.

## Baselines

Two baselines are kept separate:

1. **BM25** on the same corpus/questions for retrieval quality.
2. **No-retrieval generator**: the same deterministic extractive generator receives no evidence and therefore refuses.

This separation avoids letting generation hide retrieval errors.

## Retrieval results

| Metric | Hashing vector | BM25 baseline |
| --- | ---: | ---: |
| MRR | 0.833 | 0.917 |
| Hit Rate@1 | 0.833 | 0.833 |
| Recall@1 | 0.833 | 0.833 |
| Recall@3 | 0.833 | 1.000 |
| nDCG@3 | 0.833 | 0.938 |

The hashing vector retriever is intentionally a mechanics baseline, not a semantic model. BM25 is stronger at top-3 on this benchmark; this is evidence that changing representation from sparse to dense alone does not create semantic understanding.

## Generation / grounding results

| Metric | Naive RAG |
| --- | ---: |
| Answer contains reference | 0.833 |
| Mean token F1 | 0.391 |
| Grounded token recall | 1.000 |
| Citation precision | 0.833 |
| Citation recall | 0.833 |
| No-retrieval answer contains reference | 0.000 |

The extractive generator returns the top retrieved chunk verbatim. That makes groundedness easy to inspect and deliberately prevents a language model from masking a retrieval miss with memorized knowledge. Token F1 is modest because the generator returns a full evidence sentence rather than a concise answer; `answer_contains_reference` is the more appropriate deterministic correctness metric for this M01 generator.

## System measurements

The runner records embedding, retrieval, generation, and end-to-end latency plus approximate whitespace-token counts. On the recorded local run, end-to-end latency was roughly 0.276 ms/query and retrieval roughly 0.245 ms/query. These values are sanity/regression measurements only; this corpus is far too small for performance conclusions.

## Failure analysis

`qa6` preserves the difficult paraphrase failure:

```text
question:  How can conceptual likeness between paraphrases be retrieved?
relevant:  d5
hash top3: d8, d2, d4
answer:    evidence from d8 (cosine similarity)
```

The answer is **fully grounded but wrong**: grounded token recall remains 1.0 while citation precision/recall and answer correctness drop to 0 for this query. This demonstrates why groundedness and correctness must be evaluated separately.

Classification: **retrieval miss caused by embedding representation lacking learned semantics**.

BM25 retrieves the relevant document within top-3 for this question because of incidental lexical overlap, while the hashing vector representation does not. M02 therefore needs a real semantic dense retriever and must test whether it fixes this query without losing exact-term behavior.

## What M01 establishes

- explicit `Document` / `Chunk` / retrieved-result data models
- fixed-size chunking with overlap support
- swappable embedding and generator interfaces
- deterministic feature-hashing embeddings
- minimal in-memory cosine vector index
- context/prompt construction
- chunk-level citations
- end-to-end trace with latency/token measurements
- retrieval evaluation independent from answer evaluation
- deterministic answer correctness, groundedness, and citation metrics

## Conclusion

M01 satisfies the pipeline/evaluation learning objective, but it intentionally **does not claim semantic dense retrieval**. The key result is architectural and evaluative: the end-to-end RAG path is observable, measurable, and capable of exposing the important case “grounded but wrong because retrieval failed.”

The next milestone is M02 — Retrieval Families, starting with a real neural semantic embedding baseline against BM25 and the current hashing-vector implementation.
