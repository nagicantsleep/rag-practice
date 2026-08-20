# M02 Full Pretrained SPLADE Checkpoint

Experiment: `m02_splade_checkpoint_v1`  
Model: `naver/splade-v3-distilbert`  
Resolved Hugging Face revision: `2db06b86d65e316e2ca9907aa1aa8be6f8c4e739`  
Sentence Transformers: `5.6.1`

This is a real pretrained SPLADE-family SparseEncoder checkpoint, unlike the earlier mechanism-only linear sparse model.

| Metric | All | Exact | Semantic |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.900 | 1.000 | 0.800 |
| Recall@3 | 1.000 | 1.000 | 1.000 |
| MRR | 0.950 | 1.000 | 0.900 |

## Sparse footprint

- total non-zero document values: 1832
- mean non-zero values/document: 183.2
- document encoding: 337.7 ms
- mean query scoring: 52.88 ms

## Top-1 failures

- `s1`: got `d8`, expected one of `d5`


Timings are GitHub Actions CPU sanity measurements. The corpus is tiny, so quality conclusions are benchmark-specific.
