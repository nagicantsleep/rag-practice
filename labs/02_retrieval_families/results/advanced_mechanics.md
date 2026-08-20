# M02 Advanced Mechanics — Learned Sparse and Late Interaction

Experiment: `m02_advanced_mechanics_v1`  
Benchmark: `benchmarks/m02_retrieval@v1`  
Tests after implementation: **14 passed**

## Scope warning

These implementations teach the mechanisms of two retrieval families; they are **not checkpoint reproductions**.

- `LearnedSparseRetriever` is **SPLADE-style**: explicit vocabulary dimensions, learned lexical expansion, non-negative `log(1 + ReLU(x))` weights, dot-product retrieval, and sparsity pressure. It replaces SPLADE's pretrained Transformer/MLM backbone with a transparent bag-of-words expansion layer.
- `LateInteractionRetriever` is **ColBERT-style**: independently encoded query/document token vectors with token-level MaxSim followed by a sum over query tokens. It replaces contextual BERT with small learned token embedding tables and does not implement ColBERTv2 residual compression.

The distinction is intentional: if a simplified mechanism performs poorly, the report must not attribute that result to the full research system.

## Hypothesis

Learned sparse representations should expose interpretable lexical expansion while remaining sparse. Late interaction should retain finer token-level matching than a single-vector retriever, at the cost of storing multiple vectors per document.

## Hyperparameters and evaluation discipline

The learned-sparse sparsity coefficient was selected using the existing dev split; `lambda = 0.01` gave the strongest dev ranking among the tested sparsity settings while keeping document representations sparse. The late-interaction training length was shortened once additional epochs did not improve dev quality. The held-out test metrics below use the same M02 test set as the core sub-lab.

## Results

| Method | All R@1 | Exact R@1 | Semantic R@1 | All R@3 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 core baseline | 0.800 | 1.000 | 0.600 | 0.900 | 0.850 |
| Neural single-vector core | 0.800 | 0.800 | 0.800 | 1.000 | 0.900 |
| Weighted hybrid core | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |
| SPLADE-style mechanics | 0.600 | 0.800 | 0.400 | 0.800 | 0.734 |
| ColBERT-style mechanics | 0.700 | 0.800 | 0.600 | 0.800 | 0.775 |

Neither simplified advanced model beats the core baselines. That is a useful result, not something to hide: **the research method is more than its final scoring formula**. SPLADE relies on a pretrained masked-language-model backbone and carefully designed sparse training; ColBERT relies on contextual token encodings, and ColBERTv2 additionally addresses storage and supervision.

## Learned sparse: interpretability and sparsity

The vocabulary has 164 dimensions, but the learned document representation averages only **4 non-zero dimensions** at the recorded threshold. This demonstrates an explicit sparse representation rather than a dense vector with many zeros hidden behind an ANN abstraction.

A successful semantic query is:

```text
query: orientation of numerical representations
relevant: d8 (cosine similarity / vectors)
```

The learned query representation includes readable expansion dimensions:

```text
representations  1.024
vector           0.578
similarity       0.495
```

The original query does not contain `vector` or `similarity`, so this is direct evidence of **learned lexical expansion**. The same model still misses the preserved hard paraphrase `s1`; with no pretrained language backbone, its expansion generalizes poorly beyond the small supervised set.

## Late interaction: MaxSim and footprint

The late-interaction model stores a vector for each document token rather than one vector per document. On this tiny corpus it stores **162 token vectors × 16 dimensions**, compared with the core neural dual encoder's **10 document vectors × 32 dimensions**.

It retrieves the preserved paraphrase `s1` correctly and also retrieves `s2` correctly at rank 1, but it regresses other queries such as `e3`, `s3`, and `s5`.

This makes the representation trade-off concrete:

```text
single-vector dense:  compact document representation, early interaction lost
late interaction:     fine token-level MaxSim, much larger index representation
learned sparse:       explicit vocabulary weights, inverted-index-friendly sparsity
```

## System observations

On the recorded CPU run, learned-sparse training took about 1.15 s and late-interaction training about 0.20 s after vectorizing the MaxSim batch computation. Query latencies are sub-millisecond only because the corpus has 10 documents; no scalability claim is made.

## Failure analysis

The largest lesson is **backbone/contextualization failure**, not “SPLADE is bad” or “ColBERT is bad.” The simplified models have no broad pretrained semantics. Unseen terms collapse toward `<unk>`, and static token embeddings cannot represent context-dependent meaning.

The benchmark is also too small to evaluate real indexing efficiency. A future checkpoint reproduction must measure true sparse postings / token-vector storage on a larger corpus.

## Conclusion

This sub-lab satisfies its mechanism-learning objective and evaluation gate. It makes sparse expansion and late interaction inspectable, exposes their representation footprints, and produces explicit negative evidence showing why pretrained contextual backbones matter.

M02 remains `IN PROGRESS`: full pretrained dense, SPLADE checkpoint, and ColBERT/ColBERTv2 checkpoint evaluation are still open because no pretrained model artifacts are currently available in the execution environment.
