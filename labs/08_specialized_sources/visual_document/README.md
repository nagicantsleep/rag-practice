# M08.6 — Visual-document / page-image RAG

Status: **EVIDENCE RECORDED — FINAL GATE PENDING**

## Goal

Test the boundary between OCR/text-extraction evidence and page-image/layout evidence without treating one as equivalent to the other.

The frozen benchmark contains 6 synthetic page rasters and 10 queries covering text facts, visual layout, a highlighted table cell, a chart comparison, cross-modal document identity, region provenance, and no-evidence cases.

## Systems

1. `ocr_surrogate` — BM25 over frozen OCR text only. It never inspects page pixels.
2. `page_native` — deterministic raster/layout control over the exact frozen page image. It never reads OCR facts.
3. `ocr_page_fusion` — explicit document constraints plus OCR relevance and raster evidence, preserving both modalities and page/region locators.
4. `pinned_colsmol` — exhaustive pretrained text-to-page-image retrieval using `vidore/colSmol-256M` adapter revision `a59110fdf114638b8018e6c9a018907e12f14855` over `vidore/ColSmolVLM-Instruct-256M-base` revision `8a0cee6d479200dbce31dbfef88c66175d89cddc`. Ranking receives query text and rendered page pixels only.

## Results

| System | Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Region locator | Visual candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OCR surrogate | 0.875 | 0.750 | 0.667 | **1.000** | **1.000** | 0.500 | 0.400 | 0.000 | 0.000 | **0.0** |
| page-native control | 0.750 | 0.750 | **1.000** | **1.000** | 0.000 | **1.000** | 0.700 | **1.000** | **1.000** | 4.2 |
| OCR + page fusion | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 3.2 |
| pinned ColSmol | **1.000** | 0.875 | **1.000** | **1.000** | 0.500 | 0.000 | 0.400 | **1.000** | 0.000 | 6.0 |

Pinned ColSmol CPU/float32 evidence records model load `4508.89 ms`, six-page index build `136884.19 ms`, embedding shape `[6, 875, 128]` / `2,688,000` logical bytes, and mean query latency `100.73 ms`. These are GitHub Actions CPU sanity measurements, not serving benchmarks.

## Findings

- **OCR page hits are not visual evidence.** OCR retrieval reaches text Hit@1 `1.0`, but visual grounding and region provenance remain `0.0` because the raster is never inspected.
- **Pixels can identify layout while missing text facts.** The deterministic page-native control reaches visual Hit@1/grounding/region accuracy `1.0`, while text-sufficient Hit@1 is `0.0` because frozen OCR facts are unavailable.
- **Explicit fusion is the controlled mechanism ceiling, not a general document-AI claim.** Perfect scores come from transparent OCR plus exact deterministic raster features on a tiny synthetic corpus.
- **The pinned pretrained control is useful but incomplete.** ColSmol retrieves every relevant page within top-3 (`Recall@3 1.0`) and solves every frozen visual/cross-modal page-selection query at rank 1, but it puts the Alpha appendix above the Alpha operations page for the hotline query, giving overall Hit@1 `0.875` and text-sufficient Hit@1 `0.5`.
- **Page retrieval is not answer extraction.** The pretrained control is deliberately not allowed to borrow frozen OCR after ranking, so text/table value answers remain unsupported and answer correctness is `0.4` even when the right page is retrieved.
- **A page retriever is not a region localizer.** ColSmol supplies page-level multi-vector similarity but this control exposes no region locator, so region provenance remains `0.0`; no deterministic locator heuristic is added after ranking.
- **An exhaustive retriever is not an abstention policy.** ColSmol always ranks all six pages and therefore scores `0.0` on both frozen no-evidence cases. The false positives are retained rather than calibrated on the test set.
- **Mutable upstream checkpoints are a reproducibility risk.** The adapter's referenced base repository default revision no longer exposed the full weights during CI. The control therefore pins both the adapter and the verified historical full-weight base revision.

## Benchmark integrity

The benchmark was frozen before successful pretrained inference. CI exposed repository-transport defects in the raster bundle (JSON envelope and one gzip trailer). Repairs changed only the transport envelope: recovered XPM bytes were rendered to RGB `128×176` and verified against the six already-frozen SHA-256 RGB hashes before the pretrained model produced a result. Queries, qrels, expected answers, OCR text, region labels, and rendered pixels were not changed to fit ColSmol behavior.

## Evidence

- Phase-1 repaired-benchmark gate `32474824392` / job `96748883250`: **116 tests passed** and deterministic evaluation passed; the then-unpinned upstream base lookup failed before pretrained inference.
- Full pinned pretrained PR gate `32475115855` / job `96749747685`: **116 tests passed**, deterministic evaluation passed, and pinned ColSmol evaluation passed.
- Deterministic and pretrained JSON/Markdown evidence was persisted by the push workflow in commit `a263bc29c43bd2921c49aeb0958c9a582af2da61`.

## Definition of Done

- [x] Freeze page records, raster payloads, OCR surrogates, queries, qrels, answers, and region labels before pretrained inspection.
- [x] Keep OCR evidence and page-raster evidence explicitly attributable.
- [x] Record page locators and region locators separately from answer correctness.
- [x] Evaluate page retrieval, visual grounding, region provenance, abstention, latency, and representation footprint separately.
- [x] Add a pinned pretrained page-image / visual-document retrieval control on the same frozen benchmark.
- [x] Persist pretrained results and retain negative/error cases without tuning the frozen benchmark around them.
- [ ] Pass the final source-of-truth full-regression + deterministic + pretrained evaluation gate on the findings head.
- [ ] Record final completion in ROADMAP and mark M08.6 complete.

## Guardrails

The deterministic raster control is a teaching mechanism, not a general visual-document model. A lexical page hit can still have zero visual grounding. A page-image hit can still be unable to recover OCR-only facts. An exhaustive pretrained retriever is not automatically an abstention policy.
