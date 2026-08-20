# M00 — Information Retrieval Fundamentals

## Learning objective

Implement the retrieval math that later RAG systems depend on before introducing embedding APIs, vector databases, orchestration frameworks, or LLM generation.

## Hypothesis

On a small lexical benchmark, TF-IDF/cosine and BM25 should retrieve exact-term and near-exact-term evidence reliably. Both should struggle when a relevant document shares little or no vocabulary with the query. That failure is intentional: it motivates dense semantic retrieval in later milestones.

## Implemented

- deterministic tokenizer
- explicit inverted index and collection statistics
- dense-vector cosine similarity helper
- sparse-vector cosine similarity
- TF / smoothed IDF / TF-IDF vectors
- TF-IDF cosine retrieval
- BM25 implemented from the formula
- top-k search
- Hit Rate@K
- Precision@K
- Recall@K
- reciprocal rank / MRR
- average precision / MAP
- DCG / nDCG@K
- deterministic relevance judgments (`qrels`)
- query-level retrieval failure capture

## Benchmark

The benchmark is intentionally tiny and inspectable:

- `benchmarks/m00_ir/corpus.jsonl` — 10 short documents
- `benchmarks/m00_ir/queries.jsonl` — 10 queries
- `benchmarks/m00_ir/qrels.json` — binary/graded relevance judgments

`q10` deliberately describes dense semantic retrieval without using vocabulary that appears in the relevant document. A lexical method should miss it, giving us a known failure case rather than a benchmark where every baseline is perfect.

## Run

```bash
PYTHONPATH=src python labs/00_ir_fundamentals/run.py
```

Run the automated checks with:

```bash
pytest
```

## Evaluation protocol

Compare TF-IDF/cosine and BM25 with the same corpus, queries, relevance labels, and `top_k=5`.

Primary retrieval metrics:

- MRR
- MAP
- Recall@1/3/5
- Precision@1/3/5
- Hit Rate@1/3/5
- nDCG@1/3/5

System measurement:

- mean retrieval latency per query

The latency number is educational only: this corpus is far too small for performance conclusions.

## Metric verification

`tests/test_retrieval_metrics.py` includes hand-computable rankings so metric code is validated independently of the retrievers. This is important because all later milestones will reuse these evaluation utilities.

## Findings

See `results/baseline.md` and `results/baseline.json`.

## Definition of Done

- [x] learning objective is written
- [x] algorithm/pipeline is implemented
- [x] core behavior has automated tests
- [x] baseline comparison is runnable
- [x] benchmark/evaluation dataset is defined
- [x] retrieval evaluation is implemented
- [x] latency is recorded
- [x] results are saved
- [x] failure cases are inspected
- [x] findings and trade-offs are written down

M00 intentionally has no answer-generation evaluation because there is no generator yet. Generation/groundedness evaluation becomes mandatory starting with M01.
