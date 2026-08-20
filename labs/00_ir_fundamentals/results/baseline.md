# M00 Baseline Results — TF-IDF vs BM25

Experiment: `m00_lexical_baselines_v1`

Benchmark: `benchmarks/m00_ir@v1`  
Corpus: 10 documents  
Queries: 10  
Retrieval cutoff: `k=5`

## Aggregate retrieval quality

| Metric | TF-IDF + cosine | BM25 |
| --- | ---: | ---: |
| MRR | 0.900 | 0.900 |
| MAP | 0.900 | 0.900 |
| Hit Rate@1 | 0.900 | 0.900 |
| Recall@1 | 0.750 | 0.750 |
| nDCG@1 | 0.900 | 0.900 |
| Hit Rate@3 | 0.900 | 0.900 |
| Recall@3 | 0.900 | 0.900 |
| nDCG@3 | 0.900 | 0.900 |
| Hit Rate@5 | 0.900 | 0.900 |
| Recall@5 | 0.900 | 0.900 |
| nDCG@5 | 0.900 | 0.900 |

On this deliberately small benchmark, the two lexical methods have identical relevance metrics. This is not evidence that the algorithms are equivalent; it means the current corpus is too small/simple to expose ranking differences beyond their shared lexical limitation.

## Failure analysis

Both methods fail `q10`:

```text
query:     conceptual likeness between paraphrases
relevant:  d5
retrieved: d8
```

Relevant `d5` describes dense retrieval using terms such as `embeddings`, `semantically similar`, and `exact words differ`. The query expresses the same concept using `conceptual likeness` and `paraphrases`.

Neither TF-IDF nor BM25 has a semantic representation that can connect those expressions. The only lexical overlap found in the collection is `between`, which occurs in `d8`, so both lexical retrievers return the wrong document.

Classification: **retrieval miss caused by lexical vocabulary mismatch**.

This is the key transition point toward dense retrieval: M01/M02 should test whether embeddings recover `d5` for this query without sacrificing the strong exact-term behavior of lexical search.

## Multi-relevance observation

Some queries have more than one relevant document with graded relevance. That is why mean Recall@1 is 0.750 even though Hit Rate@1 is 0.900: a top result can be relevant while still failing to retrieve all relevant evidence at rank 1.

This distinction will matter later when a RAG answer needs multiple pieces of evidence rather than merely one relevant chunk.

## Latency

`baseline.json` records mean retrieval latency from the machine that generated the artifact. At this corpus size, these values are only a sanity check that latency is being measured. They must not be used to conclude that BM25 or TF-IDF is generally faster.

## Conclusion

M00 establishes three reusable foundations:

1. retrieval implementations are inspectable rather than framework-hidden;
2. retrieval quality is measured independently from generation;
3. failure cases are treated as learning signals and become hypotheses for the next milestone.

The next milestone should keep this benchmark and add a minimal dense RAG pipeline plus generation/groundedness evaluation.
