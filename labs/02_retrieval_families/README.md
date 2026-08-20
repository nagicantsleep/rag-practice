# Lab 02 — Retrieval Families

M02 compares retrieval families under one evaluation harness instead of treating “RAG” as a single retriever.

## Core sub-lab

Implemented and evaluated:

- BM25 lexical baseline
- hashing-vector mechanics baseline
- supervised neural dual encoder trained from scratch
- Reciprocal Rank Fusion (RRF)
- min-max normalized weighted sparse+dense fusion
- train/dev/test separation
- exact-vs-semantic query-class evaluation

The tiny dual encoder makes learned dense retrieval visible: queries and documents are independently projected into a shared dense space and trained so relevant pairs outrank negatives.

## Advanced mechanics sub-lab

Also implemented:

- **SPLADE-style learned sparse expansion** with explicit vocabulary coordinates, log-saturated non-negative weights, sparsity pressure, and inspectable expanded terms
- **ColBERT-style late interaction** with independent token vectors and MaxSim scoring
- representation-footprint measurements

These are mechanism-focused educational implementations, not pretrained checkpoint reproductions. The result report explicitly separates conclusions about the simplified models from conclusions about the research systems.

## Run

```bash
pip install -e ".[dev,neural]"
python labs/02_retrieval_families/run.py
python labs/02_retrieval_families/run_advanced_mechanics.py
pytest
```

Results:

- `results/core.json` / `results/core.md`
- `results/advanced_mechanics.json` / `results/advanced_mechanics.md`

## Evaluation contract

**Hypothesis:** learned representations should improve semantic matching; lexical retrieval remains strong on exact terms; hybrid retrieval can combine signals; learned sparse and late-interaction mechanisms introduce distinct interpretability/index-footprint trade-offs.

**Baselines:** BM25, hashing-vector, and the core neural single-vector retriever.

**Splits:** supervised training pairs, separate dev queries for hyperparameter/fusion selection, and held-out exact/semantic test queries.

**Metrics:** Recall@1/@3, MRR, MAP, nDCG, query-class breakdown, training loss, latency sanity measurements, and representation footprint.

## Canonical references

- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020): https://arxiv.org/abs/2004.04906
- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009): https://doi.org/10.1145/1571941.1572114
- Formal et al., *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval* (2021): https://arxiv.org/abs/2109.10086
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (2020): https://arxiv.org/abs/2004.12832
- Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction* (2021/2022): https://arxiv.org/abs/2112.01488

## Remaining M02 work

- pretrained general-purpose semantic embedding baseline
- full pretrained SPLADE checkpoint evaluation
- full pretrained ColBERT/ColBERTv2 checkpoint evaluation
- larger benchmark / index-size comparison

The current execution environment has PyTorch but no `transformers`, `sentence-transformers`, cached Hugging Face model, or embedding API credential. Therefore those checkpoint sub-labs remain open rather than receiving fabricated evaluation results.

M02 stays `IN PROGRESS` until the remaining checkpoint families are evaluated or the roadmap explicitly re-scopes them with evidence.
