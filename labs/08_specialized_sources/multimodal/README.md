# M08.5 — Multimodal RAG

Status: **IN PROGRESS** — low-level multimodal mechanism candidate pending CI evidence.

## Hypothesis

Text surrogates and image pixels are complementary evidence. Captions/metadata can identify an asset but cannot justify a visual claim when the relevant state is omitted; pixel-only matching can recover visual state yet lose asset identity and other textual constraints. A multimodal RAG path must keep both evidence modalities explicit rather than treating captions, OCR, and pixels as interchangeable.

## Phase 1 mechanism

The benchmark uses dependency-free ASCII P3 PPM raster images so image evidence is real pixel data while the implementation stays fully inspectable. Three systems share the same 9 assets and 10 queries:

1. **text surrogate BM25** over title/caption/site/kind/quarter, never reading pixels;
2. **pixel-native control** using exact color, quadrant, and relative bar-height features, deliberately ignoring asset metadata;
3. **multimodal fusion** using explicit site/kind constraints plus BM25 surrogate relevance and pixel evidence when the query contains a visual requirement.

Answers are extracted separately from retrieval. Text-only mode refuses unsupported visual claims, while pixel-capable modes read the selected raster. Every image retains an `image://benchmark/<file>.ppm` locator through the shared M08 `Source` contract.

## Controlled benchmark

`benchmarks/m08_multimodal/` contains 9 images and 10 queries:

- 4 visual-only questions whose captions intentionally omit the decisive color/position/bar-height state;
- 2 text-sufficient questions;
- 2 cross-modal questions requiring asset identity plus pixel evidence;
- 2 no-evidence questions testing abstention.

Paired control-panel captions are intentionally similar so a text retriever cannot infer visual state from alt text. Pixel-only retrieval is also intentionally blind to `alpha`/`beta` identity, making the cross-modal distinction observable.

## Evaluation contract

Retrieval is separated from answer quality:
- Recall@3 and Hit@1 on non-empty qrels;
- visual-required, cross-modal, and text-sufficient Hit@1;
- no-evidence accuracy;
- visual-evidence-grounded rate;
- answer correctness;
- visual candidates scored and query latency.

A short answer matching by accident does not repair a wrong retrieval trace.

## Research context and next control

CLIP learns aligned image/text representations through contrastive image-text training (arXiv:2103.00020). Recent Visual-RAG and M2RAG benchmarks emphasize image evidence rather than text-only augmentation (arXiv:2502.16636, arXiv:2502.17297). Phase 2 will add a pinned pretrained CLIP text-to-image retrieval control without changing this frozen benchmark. ColPali/page-image retrieval is reserved for the later visual-document sub-lab.

## Definition of Done

- [x] raster benchmark with hidden visual state defined
- [x] text-surrogate BM25 control implemented
- [x] dependency-free pixel-native control implemented
- [x] multimodal fusion and no-evidence abstention implemented
- [x] retrieval, visual grounding, answer, and candidate-cost metrics separated
- [x] shared `Source` image provenance exposed
- [x] regression tests added
- [ ] full repository CI + Phase 1 evaluator passes
- [ ] persisted Phase 1 results and failures reviewed
- [ ] pinned pretrained CLIP retrieval control evaluated on the same benchmark
- [ ] CLIP result retained even if it underperforms the handcrafted control
- [ ] final findings/error analysis written down
- [ ] ROADMAP marks Multimodal RAG DONE only after final source-of-truth gate

This phase is a controlled mechanism study, not a claim that handcrafted color/layout features are a general vision model.
