# Lab 03 — Indexing and Chunking

M03 isolates index representation choices from retriever choice. Phase 1 fixes BM25 scoring and compares chunk boundaries/metadata under one benchmark.

## Phase 1 strategies

- fixed 24-word chunks, no overlap
- fixed 24-word chunks with 8-word overlap
- sentence-aware packing
- paragraph-aware packing
- deterministic sentence-boundary similarity chunking
- sentence-aware chunks enriched with document metadata

## Metrics

- document Hit@1/@3
- evidence completeness@1/@3
- source-token utilization@3 (penalizes duplicated overlap and metadata overhead)
- relevant-context fraction@3
- chunk count / mean chunk words
- index-build and mean query latency sanity measurements

## Run

```bash
pip install -e ".[dev]"
pytest -q
python labs/03_indexing_chunking/run.py
```

Phase 2 will add parent-child and hierarchical retrieval after the boundary/metadata baselines are measured.
