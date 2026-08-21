# M08.6 frozen visual-document benchmark

This benchmark is frozen before any pretrained visual-document retrieval result is inspected.

## Corpus

Six synthetic document pages are stored as exact PNG byte payloads encoded in `images.json`.
The encoding is only repository-friendly transport; evaluators decode the bytes and inspect the rendered page image.
Each record also contains a frozen `ocr_text` surrogate. OCR text is deliberately separate from the raster and is never produced from the image at evaluation time.

The pages cover:
- text-only facts (hotline and shipping SLA),
- a colored approval stamp,
- a left sidebar,
- a highlighted table cell,
- a two-bar chart,
- and a highlighted invoice total.

## Queries

Ten frozen queries cover text-sufficient, visual-layout, table-layout, chart-layout, cross-modal, region-localization, and no-evidence cases.
`relevant` contains page-level qrels. Queries with `region` also require the returned region locator to match.

## Integrity rules

- Do not change records, qrels, expected answers, visual marker colors, page layout, or OCR text after observing pretrained model behavior.
- OCR evidence and page-raster evidence must remain separately attributable.
- A correct short answer is not visual grounding unless the evidence modality includes the page image.
- The deterministic raster control is a mechanism test, not a learned visual-document benchmark claim.
- Pretrained controls must pin model/checkpoint and runtime dependency versions and report failure modes rather than tuning this benchmark around them.
