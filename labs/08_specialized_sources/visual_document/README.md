# M08.6 — Visual-document / page-image RAG

Status: **PHASE 1 FROZEN**

## Goal

Test the boundary between OCR/text-extraction evidence and page-image/layout evidence without treating one as equivalent to the other.

The frozen benchmark contains 6 synthetic page rasters and 10 queries covering text facts, visual layout, a highlighted table cell, a chart comparison, cross-modal document identity, region provenance, and no-evidence cases.

## Systems

1. `ocr_surrogate` — BM25 over frozen OCR text only. It never inspects page pixels.
2. `page_native` — deterministic raster/layout control over the exact frozen page image. It never reads OCR facts.
3. `ocr_page_fusion` — explicit document constraints plus OCR relevance and raster evidence, preserving both modalities and page/region locators.
4. Pinned pretrained visual-document retrieval control — pending Phase 2; the benchmark is frozen before its first result.

## Definition of Done

- [x] Freeze page records, raster payloads, OCR surrogates, queries, qrels, answers, and region labels before pretrained inspection.
- [x] Keep OCR evidence and page-raster evidence explicitly attributable.
- [x] Record page locators and region locators separately from answer correctness.
- [x] Evaluate page retrieval, visual grounding, region provenance, abstention, latency, and representation footprint separately.
- [ ] Add a pinned pretrained page-image / visual-document retrieval control on the same frozen benchmark.
- [ ] Persist pretrained results and retain negative/error cases without tuning the frozen benchmark around them.
- [ ] Pass the final source-of-truth full-regression + deterministic + pretrained evaluation gate.
- [ ] Record final findings in ROADMAP and mark M08.6 complete.

## Guardrails

The deterministic raster control is a teaching mechanism, not a general visual-document model. A lexical page hit can still have zero visual grounding. A page-image hit can still be unable to recover OCR-only facts. An exhaustive pretrained retriever is not automatically an abstention policy.
