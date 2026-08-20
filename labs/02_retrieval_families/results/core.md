# M02 Core Results — Sparse, Learned Dense, and Hybrid Retrieval

Experiment: `m02_retrieval_core_v1`  
Corpus: `benchmarks/m00_ir@v1` (10 documents)  
Training pairs: 21  
Dev queries for hybrid tuning: 6  
Held-out test queries: 10 (5 exact, 5 semantic)

## Hypothesis

A **learned** dual encoder should outperform lexical/hash representations on semantic paraphrases, but may trade away some exact-term reliability. A hybrid retriever should recover part of that lexical strength while retaining semantic gains.

## Methods

- **BM25** — lexical baseline from M00.
- **Hashing vector** — M01 dense-storage mechanics baseline; lexical features hashed into a dense vector, with no learned semantics.
- **Tiny neural dual encoder** — separate query/document projections trained with full-corpus contrastive softmax. It uses bag-of-words inputs, `tanh`, L2-normalized 32-dimensional outputs, and 21 supervised training pairs.
- **Hybrid RRF** — BM25 + neural rankings fused with Reciprocal Rank Fusion.
- **Hybrid weighted score fusion** — min-max normalizes BM25 and neural scores independently, then combines them. The BM25 weight is selected on the dev set only.

The tiny dual encoder is deliberately domain-specific and is **not** presented as a pretrained general-purpose embedding model. Its purpose is to make learned dense retrieval mechanics inspectable before adding external pretrained models.

## Train / dev / test separation

Training pairs live in `train.jsonl`; hybrid weight selection uses `dev.jsonl`; final metrics below are computed on `queries.jsonl`. The held-out test set is not used to choose the fusion weight.

The neural training loss fell from **3.6830** to **0.000686** over 400 epochs.

Dev tuning selected:

```text
BM25 weight   = 0.1
neural weight = 0.9
```

## Held-out retrieval results

| Method | Recall@1 all | Exact Recall@1 | Semantic Recall@1 | Recall@3 all | MRR all |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.800 | 1.000 | 0.600 | 0.900 | 0.850 |
| Hashing vector | 0.800 | 1.000 | 0.600 | 0.800 | 0.835 |
| Neural dual encoder | 0.800 | 0.800 | 0.800 | 1.000 | 0.900 |
| Hybrid RRF | 0.800 | 0.800 | 0.800 | 1.000 | 0.900 |
| Hybrid weighted | 0.900 | 1.000 | 0.800 | 1.000 | 0.950 |

## Acceptance target: preserved paraphrase failure

The M00/M01 failure becomes test query `s1`:

```text
query:    conceptual likeness between paraphrases
relevant: d5 (dense retrieval / semantic similarity)
```

Top-1 results:

```text
BM25             d8   ✗
Hashing vector   d2   ✗
Neural encoder   d5   ✓
Hybrid RRF       d8   ✗
Hybrid weighted  d5   ✓
```

This is the first direct evidence in the repository that **learning the representation**, rather than merely storing vectors, can bridge a vocabulary mismatch.

## Important regressions

### Exact-query regression (`e3`)

BM25 and hashing retrieve `d6` correctly. The neural retriever ranks `d10` first, and plain RRF follows that error. Dev-tuned weighted fusion restores `d6`.

This is why dense retrieval should not be assumed to dominate lexical retrieval uniformly.

### Semantic-query regression (`s5`)

For `unsupported claims from a predictive text system`, BM25 retrieves `d10`, while the tiny neural model ranks `d9`. Weighted hybrid also follows the neural mistake because the tuned neural weight is high.

Classification: **learned representation/domain coverage failure**. The model has only 21 supervised pairs and no broad pretrained language knowledge.

## RRF vs score fusion

RRF is attractive because it does not require calibrated scores, but on this benchmark its top-1 result does not improve over the neural retriever. Its Recall@3 is still 1.0.

Weighted fusion performs better at top-1 after tuning on a separate dev set. This does **not** prove weighted fusion is generally superior; it demonstrates that fusion strategy and calibration are experimental variables that require evaluation.

## System observations

Training the tiny dual encoder took about **919 ms** on the recorded local CPU run. Mean query-time sanity measurements were roughly 0.017 ms for BM25, 0.282 ms for hashing, 0.132 ms for neural search, and under 0.02 ms for fusion alone on this 10-document corpus. These are not scalability claims.

## Findings

1. Dense **storage** is not semantic; M01 hashing remains a lexical-feature representation.
2. A learned dual encoder fixes the target paraphrase and improves semantic Recall@1 from 0.6 to 0.8.
3. Learned dense retrieval can regress exact queries; BM25 remains a strong complementary signal.
4. Fusion is not automatically better: plain RRF did not improve top-1 here.
5. Dev-tuned weighted hybrid achieved the best held-out overall Recall@1 (0.9) while preserving exact Recall@1 (1.0).
6. The tiny supervised model remains domain-limited; a pretrained semantic encoder is still an important next sub-lab.

## Status

This completes the **core M02 sub-lab** (BM25, dense mechanics baseline, learned dense, RRF, weighted hybrid). M02 remains `IN PROGRESS` because the roadmap still includes pretrained dense retrieval plus advanced learned-sparse/late-interaction sub-labs such as SPLADE and ColBERT.
