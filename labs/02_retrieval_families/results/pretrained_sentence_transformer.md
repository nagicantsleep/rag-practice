# M02 Pretrained Dense Baseline — all-MiniLM-L6-v2

Experiment: `m02_pretrained_minilm_v1`  
Model: `sentence-transformers/all-MiniLM-L6-v2`  
Pinned revision: `1c82ace116a2629de82404c4be48c0e5d4cf08be`  
Sentence Transformers: `5.6.1`

## Purpose

Evaluate a broadly pretrained semantic sentence encoder under the same M02 ranking harness used by BM25, hashing vectors, and the tiny supervised dual encoder.

## Retrieval quality

| Metric | All | Exact | Semantic |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.900 | 1.000 | 0.800 |
| Recall@3 | 0.900 | 1.000 | 0.800 |
| MRR | 0.925 | 1.000 | 0.850 |
| nDCG@3 | 0.900 | 1.000 | 0.800 |

## Preserved paraphrase acceptance case

`s1` top-1: `d5`; relevant: `d5`; fixed: **true**.

## System sanity measurements

- model load: 3623.4 ms
- 10-document index encoding: 53.0 ms
- mean query retrieval: 9.96 ms
- logical float32 embedding payload: 15360 bytes

## Interpretation

The pretrained model supplies semantic knowledge learned outside this tiny benchmark; ranking remains our own cosine/dot-product implementation. Compare the result against the domain-trained-from-scratch encoder to separate broad pretraining from local supervision.

Timings are GitHub Actions CPU measurements and are regression/sanity evidence, not universal performance claims.
