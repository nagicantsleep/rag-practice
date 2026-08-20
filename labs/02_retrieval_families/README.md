# Lab 02 — Retrieval Families

M02 compares retrieval families under one evaluation harness instead of treating “RAG” as a single retriever. The milestone is complete; later milestones can reuse these retrievers as controlled baselines.

## What is implemented

### Core retrieval

- BM25 lexical baseline
- hashing-vector mechanics baseline
- supervised neural dual encoder trained from scratch
- Reciprocal Rank Fusion (RRF)
- min-max normalized weighted sparse+dense fusion
- train/dev/test separation for tuning vs final evaluation
- exact-vs-semantic query-class evaluation

### Mechanism-first advanced implementations

- SPLADE-style learned sparse expansion with explicit vocabulary coordinates, log-saturated non-negative weights, sparsity pressure, and inspectable expanded terms
- ColBERT-style late interaction with token vectors and MaxSim

These educational implementations make the equations and representation trade-offs visible. They are intentionally distinguished from the real pretrained checkpoints below.

### Real pretrained checkpoints

- `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional single-vector dense semantic retrieval
- `naver/splade-v3-distilbert` — pretrained SPLADE-family sparse retrieval via Sentence Transformers `SparseEncoder`
- `colbert-ir/colbertv2.0` — pretrained ColBERTv2 evaluated via PyLate exhaustive MaxSim over all 10 candidate documents

Every model revision is recorded in the generated result artifact. ColBERT uses exhaustive scoring here; **no PLAID/ANN latency claim is made**.

### Scaling stress test

The M02 held-out queries are kept fixed while deterministic, deliberately off-topic distractors grow the corpus from 10 to 100 to 1000 documents. This tests candidate-set robustness, index/representation growth, and simple exhaustive-search costs without pretending the synthetic distractors are a broader semantic benchmark.

## Run

Core and mechanism-first labs:

```bash
pip install -e ".[dev,neural]"
python labs/02_retrieval_families/run.py
python labs/02_retrieval_families/run_advanced_mechanics.py
pytest
```

Pretrained dense + SPLADE + scaling:

```bash
pip install -e ".[dev,neural,pretrained]"
python labs/02_retrieval_families/run_pretrained.py
python labs/02_retrieval_families/run_splade_checkpoint.py
python labs/02_retrieval_families/run_scaling.py
```

ColBERT is kept as a separate dependency step because PyLate 1.6.0 pins Sentence Transformers 5.3.0 while the `pretrained` extra pins 5.6.1:

```bash
pip install "pylate==1.6.0"
python labs/02_retrieval_families/run_colbert_checkpoint.py
```

The GitHub Actions workflow reproduces this dependency transition and is the canonical checkpoint evaluation path.

## Result artifacts

- `results/core.json` / `results/core.md`
- `results/advanced_mechanics.json` / `results/advanced_mechanics.md`
- `results/pretrained_sentence_transformer.json` / `.md`
- `results/splade_checkpoint.json` / `.md`
- `results/colbert_checkpoint.json` / `.md`
- `results/scaling.json` / `.md`
- `results/m02_summary.json` / `.md`

## Held-out summary

| Method | Recall@1 | Exact R@1 | Semantic R@1 | Recall@3 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.800 | 1.000 | 0.600 | 0.900 | 0.850 |
| Hashing vector | 0.800 | 1.000 | 0.600 | 0.800 | 0.835 |
| MiniLM pretrained dense | **0.900** | **1.000** | **0.800** | 0.900 | 0.925 |
| SPLADE pretrained sparse | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |
| ColBERTv2 late interaction | **0.900** | **1.000** | **0.800** | **1.000** | **0.950** |

Do not over-rank the checkpoint models from this table: the benchmark has only ten held-out questions. The more useful observation is that they make **different mistakes** despite similar aggregate scores.

## Evaluation contract

**Hypothesis:** learned/pretrained representations improve semantic matching; lexical retrieval remains strong for exact terms; fusion can help when signals are complementary; sparse/dense/late-interaction representations impose different interpretability and serving costs.

**Baselines:** BM25, hashing vector retrieval, and the from-scratch neural dual encoder.

**Splits:** supervised training data, a separate dev query set for fusion/hyperparameter choice, and a held-out exact/semantic test set.

**Metrics:** Recall@1/@3, MRR, MAP, nDCG, query-class breakdown, CPU latency sanity measurements, and representation/index footprint. Generation metrics are not applicable because M02 intentionally isolates retrieval.

**CI evidence:** final PR workflow run `32395089427` succeeded with **32 tests passed**, then successfully evaluated MiniLM, full SPLADE, the scaling stress test, and full ColBERTv2.

## Failure analysis

- BM25/hashing miss true vocabulary-mismatch semantic queries.
- MiniLM retrieves `d5` correctly for `s1` but misses `s2` (`evidence lookup combined with a text generator`) even in its top-3.
- SPLADE and ColBERTv2 solve `s2` at rank 1 but rank `d8` above relevant `d5` for `s1`; both recover `d5` at rank 2.
- The mechanism-only SPLADE/ColBERT implementations are weaker than their pretrained counterparts, demonstrating the importance of contextual pretraining/training recipes.
- On the pretrained MiniLM dev set, weighted hybrid tuning selects BM25 weight `0.0`; hybridization is not intrinsically beneficial.

## Canonical references

- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (2020): https://arxiv.org/abs/2004.04906
- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* (SIGIR 2009): https://doi.org/10.1145/1571941.1572114
- Formal et al., *SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval* (2021): https://arxiv.org/abs/2109.10086
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (2020): https://arxiv.org/abs/2004.12832
- Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction* (2021/2022): https://arxiv.org/abs/2112.01488

## Completion conclusion

M02 is complete for the learning objective: the repo now contains lexical, dense single-vector, learned sparse, late-interaction, and hybrid retrieval families with both transparent mechanism implementations and real pretrained checkpoint evidence. Production ANN/PLAID serving belongs to M10; chunk/index construction effects begin in M03.
