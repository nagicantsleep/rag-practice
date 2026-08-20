# Lab 03 — Indexing and Chunking

M03 isolates index representation choices from retriever choice. BM25 scoring stays fixed while chunk boundaries, metadata placement, and hierarchy change.

## Phase 1 — chunk boundaries and metadata

Compared:

- fixed 24-word chunks, no overlap
- fixed 24-word chunks with 8-word overlap
- sentence-aware packing
- paragraph-aware packing
- deterministic sentence-boundary similarity chunking
- sentence-aware chunks enriched with document metadata

Held-out findings on `benchmarks/m03_chunking@v1`:

- overlap raises evidence completeness but repeats context tokens
- sentence/paragraph boundaries dramatically improve single-hit evidence completeness on this corpus
- metadata enrichment fixes the metadata-dependent Arctic/Tropical ambiguity but spends context tokens repeating metadata
- the simple hashing-similarity semantic boundary heuristic over-splits and is retained as a negative result

## Phase 2 — parent-child and hierarchy

Adds:

- sentence-child BM25 retrieval with paragraph-parent expansion
- document-level metadata+body routing followed by plain sentence-leaf retrieval

The second design tests whether metadata can influence routing without being copied into every returned context chunk.

## Metrics

- document Hit@1/@3
- evidence completeness@1/@3
- source-token utilization@3 (penalizes duplicated overlap and metadata context overhead)
- relevant-context fraction@3
- route Hit@1 for hierarchical retrieval
- searchable representation words
- index-build and mean query latency sanity measurements

## Run

```bash
pip install -e ".[dev,neural,pretrained]"
pytest -q
python labs/03_indexing_chunking/run.py
python labs/03_indexing_chunking/run_phase2.py
```

Results are saved under `results/phase1.*` and `results/phase2.*`.
