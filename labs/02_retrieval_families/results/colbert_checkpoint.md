# M02 Full Pretrained ColBERTv2 Checkpoint

Experiment: `m02_colbertv2_checkpoint_v1`  
Model: `colbert-ir/colbertv2.0`  
Resolved Hugging Face revision: `c1e84128e85ef755c096a95bdb06b47793b13acf`  
PyLate: `1.6.0`

This evaluates the canonical pretrained ColBERTv2 checkpoint with token-level late interaction. Every document is supplied as a candidate and reranked exhaustively; this is checkpoint/scoring evaluation, not a PLAID index latency benchmark.

| Metric | All | Exact | Semantic |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.900 | 1.000 | 0.800 |
| Recall@3 | 1.000 | 1.000 | 1.000 |
| MRR | 0.950 | 1.000 | 0.900 |

## Multi-vector footprint

- stored document token vectors: 211
- logical embedding payload: 108032 bytes
- document encoding: 704.1 ms
- mean query encode + exhaustive MaxSim rerank: 75.25 ms

## Top-1 failures

- `s1`: got `d8`, expected one of `d5`


Timings are GitHub Actions CPU sanity measurements. Production ColBERT retrieval uses a specialized multi-vector index; that serving problem is intentionally out of scope here.
