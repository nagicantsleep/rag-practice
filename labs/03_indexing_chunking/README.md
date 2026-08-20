# Lab 03 — Indexing and Chunking

M03 isolates index representation choices from retriever choice. BM25 scoring stays fixed while chunk boundaries, metadata placement, parent expansion, and hierarchy change.

## Phase 1 — chunk boundaries and metadata

Compared:

- fixed 24-word chunks, no overlap
- fixed 24-word chunks with 8-word overlap
- sentence-aware packing
- paragraph-aware packing
- deterministic sentence-boundary similarity chunking
- sentence-aware chunks enriched with document metadata

Held-out findings on `benchmarks/m03_chunking@v1`:

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Source-token utilization@3 |
| --- | ---: | ---: | ---: | ---: |
| fixed_24 | 1.000 | 0.200 | 0.800 | 1.000 |
| fixed_24_overlap_8 | 1.000 | 0.400 | 1.000 | 0.886 |
| sentence_35 | 0.800 | 0.800 | 1.000 | 1.000 |
| paragraph_80 | 0.800 | 0.800 | 1.000 | 1.000 |
| semantic_50 | 0.800 | 0.200 | 0.800 | 1.000 |
| sentence_35_metadata | 1.000 | 0.800 | 1.000 | 0.635 |

The controlled benchmark shows that overlap raises evidence coverage but repeats context tokens. Natural sentence/paragraph boundaries package evidence better than arbitrary 24-word cuts, while metadata enrichment resolves the Arctic/Tropical ambiguity at the cost of repeating metadata in returned context. The deterministic hashing-similarity boundary heuristic over-splits and remains a negative result.

## Phase 2 — parent-child and hierarchy

Adds:

- sentence-child BM25 retrieval with paragraph-parent expansion
- document-level metadata+body routing followed by plain sentence-leaf retrieval

| Strategy | Doc Hit@1 | Evidence@1 | Evidence@3 | Utilization@3 | Searchable words | Route Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_24_overlap_8 | 1.000 | 0.400 | 1.000 | 0.886 | 367 | — |
| sentence_35_metadata | 1.000 | 0.800 | 1.000 | 0.635 | 434 | — |
| parent_child | 0.800 | 0.600 | 1.000 | 1.000 | 271 | — |
| hierarchical_metadata_root | 1.000 | 1.000 | 1.000 | 1.000 | 601 | 1.000 |

Parent-child expansion increases the chance that one selected item contains complete evidence, but it cannot solve metadata-dependent routing when metadata is absent from child search text. The hierarchical design puts metadata in a document-level routing representation and returns plain sentence leaves. On this benchmark it reaches perfect route/document/evidence accuracy and perfect source-token utilization, but uses the largest searchable representation. The result is a storage/routing-versus-context-budget trade-off, not a free quality gain.

## Representative failures retained

- `q3` (Arctic versus Tropical deployment) exposes metadata placement: sentence, paragraph, semantic, and parent-child body-only search rank Tropical first; flat metadata enrichment and metadata-root hierarchy recover Arctic.
- `semantic_50` creates 19 small chunks versus 11 sentence chunks and drops Evidence@1 to `0.2`. The heuristic is intentionally retained rather than tuning the benchmark until it wins.
- fixed 24-word chunks have Doc Hit@1 `1.0` but Evidence@1 `0.2`, demonstrating that document-level hit metrics can hide incomplete retrieved evidence.

## Metrics

- document Hit@1/@3
- evidence completeness@1/@3
- source-token utilization@3 (penalizes duplicated overlap and metadata context overhead)
- relevant-context fraction@3
- route Hit@1 for hierarchical retrieval
- searchable representation words
- index-build and mean query latency sanity measurements

## Reproducibility and scope

BM25 is used at every searchable layer, so M03 changes representation/routing rather than retriever family. `benchmarks/m03_chunking@v1` is a deliberately small controlled benchmark; it is appropriate for mechanism/error analysis, not for claiming a universal ranking of chunking methods. Timings are GitHub Actions CPU sanity measurements.

Final phase-2 CI run `32407289218`: **39 tests passed**; both evaluation steps succeeded.

## Run

```bash
pip install -e ".[dev,neural,pretrained]"
pytest -q
python labs/03_indexing_chunking/run.py
python labs/03_indexing_chunking/run_phase2.py
```

Results are saved under `results/phase1.*`, `results/phase2.*`, and `results/m03_summary.*`.
