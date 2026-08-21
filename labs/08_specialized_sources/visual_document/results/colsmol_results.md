# M08.6 pinned ColSmol page-image retrieval results

Adapter: `vidore/colSmol-256M` pinned to `a59110fdf114638b8018e6c9a018907e12f14855`.
Base: `vidore/ColSmolVLM-Instruct-256M-base` pinned to `8a0cee6d479200dbce31dbfef88c66175d89cddc`.

Ranking uses query text + rendered page pixels only. OCR text, page titles/document ids, qrels, expected answers, deterministic visual markers, and region heuristics are excluded from ranking.

| Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Region locator | Visual candidates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 | 0.875 | 1.000 | 1.000 | 0.500 | 0.000 | 0.400 | 1.000 | 0.000 | 6.0 |

## Runtime / representation

- model load: 8406.14 ms
- page index build: 119798.50 ms
- embedding shape: `[6, 875, 128]`
- embedding bytes: 2688000
- mean query latency: 85.39 ms

## Guardrails

- This is a frozen tiny synthetic benchmark, not a general visual-document leaderboard claim.
- Both the adapter and its full-weight base are pinned because the adapter's upstream base default revision is mutable.
- The retriever is exhaustive and has no abstention policy; no-evidence errors are retained.
- Region locator accuracy is intentionally zero unless the pretrained retrieval control itself exposes region provenance; no deterministic region heuristic is added after ranking.
- Text/table value questions are not answered from frozen OCR after pretrained retrieval; retrieval quality and answer capability stay separate.
