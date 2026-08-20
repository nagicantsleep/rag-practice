# Lab 02 — Retrieval Families

M02 compares retrieval families under one evaluation harness instead of treating “RAG” as a single retriever.

## Core sub-lab

Implemented now:

- M00 BM25 lexical baseline
- M01 hashing-vector mechanics baseline
- a tiny supervised neural dual encoder trained from scratch with PyTorch
- Reciprocal Rank Fusion (RRF)
- min-max normalized weighted sparse+dense score fusion
- train/dev/test separation for fusion tuning
- exact-vs-semantic query-class evaluation

The tiny dual encoder follows the same high-level idea as dual-encoder dense passage retrieval: independently encode queries/documents into a shared dense space and train relevant pairs to score above negatives. It is intentionally tiny so its training loop remains visible.

## Run

```bash
pip install -e ".[dev,neural]"
python labs/02_retrieval_families/run.py
pytest
```

Results:

- `results/core.json`
- `results/core.md`

## Evaluation contract

**Hypothesis:** learned dense retrieval should improve semantic paraphrases while lexical retrieval remains strong on exact-term queries; hybrid retrieval should combine complementary signals.

**Baselines:** BM25 and hashing-vector retrieval.

**Splits:** supervised training pairs, separate dev queries for hybrid-weight selection, and held-out exact/semantic test queries.

**Metrics:** Recall@1/@3, MRR, MAP, nDCG, query-class breakdown, training loss, and local latency sanity measurements.

## Canonical references

- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020): https://arxiv.org/abs/2004.04906
- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009): https://doi.org/10.1145/1571941.1572114

## Remaining M02 work

- pretrained general-purpose semantic embedding baseline
- learned sparse retrieval (SPLADE family)
- late interaction (ColBERT/ColBERTv2)
- larger benchmark / index-size comparison

M02 stays `IN PROGRESS` until those advanced sub-labs are either implemented or explicitly scoped with evaluation evidence.
