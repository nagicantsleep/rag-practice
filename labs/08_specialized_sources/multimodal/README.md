# M08.5 — Multimodal RAG

Status: **IN PROGRESS** — deterministic controls plus pinned CLIP candidate pending final CI evidence.

## Hypothesis

Text surrogates and image pixels are complementary evidence. Captions/metadata can identify an asset but cannot justify a visual claim when the relevant state is omitted; pixel-only matching can recover visual state yet lose asset identity and other textual constraints. A multimodal RAG path must keep both evidence modalities explicit rather than treating captions, OCR, and pixels as interchangeable.

## Phase 1 mechanism

The benchmark uses dependency-free ASCII P3 PPM raster images so image evidence is real pixel data while the implementation stays fully inspectable. Three systems share the same 9 assets and 10 queries:

1. **text surrogate BM25** over title/caption/site/kind/quarter, never reading pixels;
2. **pixel-native control** using exact color, quadrant, and relative bar-height features, deliberately ignoring asset metadata;
3. **multimodal fusion** using explicit site/kind constraints plus BM25 surrogate relevance and pixel evidence when the query contains a visual requirement.

Answers are extracted separately from retrieval. Text-only mode refuses unsupported visual claims, while pixel-capable modes read the selected raster. Every image retains an `image://benchmark/<file>.ppm` locator through the shared M08 `Source` contract.

## Phase 2 pretrained control

Phase 2 adds exhaustive text-to-image retrieval with `openai/clip-vit-base-patch32` pinned to Hugging Face revision `b97b0100e55e367c057773c2a614676470b0d575`.

The CLIP control ranks from **query text + image pixels only**. Asset title, caption, site, kind, qrels, and expected answers are excluded from image embedding and ranking. The 8x8 square rasters are resized to the model image size with PyTorch bicubic interpolation and normalized with the canonical CLIP image mean/std before `CLIPModel.get_image_features`; text uses the tokenizer from the same pinned revision.

This is intentionally a retrieval-only control, not a fusion system. It therefore has no explicit no-evidence rejection policy and cannot use hidden metadata to repair cross-modal identity mistakes. Answer quality is still measured separately by applying the existing deterministic pixel reader only after CLIP chooses an image.

## Controlled benchmark

`benchmarks/m08_multimodal/` contains 9 images and 10 queries:

- 4 visual-only questions whose captions intentionally omit the decisive color/position/bar-height state;
- 2 text-sufficient questions;
- 2 cross-modal questions requiring asset identity plus pixel evidence;
- 2 no-evidence questions testing abstention.

Paired control-panel captions are intentionally similar so a text retriever cannot infer visual state from alt text. Pixel-only retrieval is also intentionally blind to `alpha`/`beta` identity, making the cross-modal distinction observable.

Before the first pretrained CLIP evaluation, two malformed ASCII PPM payloads (`p1.ppm` and `p4.ppm`) were normalized to the intended 8x8 RGB payload length without changing their intended marker color/quadrant semantics. A regression test now parses all nine frozen rasters and asserts 64 RGB pixels each. No CLIP result had been observed before this fixture repair.

## Evaluation contract

Retrieval is separated from answer quality:
- Recall@3 and Hit@1 on non-empty qrels;
- visual-required, cross-modal, and text-sufficient Hit@1;
- no-evidence accuracy;
- visual-evidence-grounded rate;
- answer correctness;
- visual candidates scored and query latency.

A short answer matching by accident does not repair a wrong retrieval trace.

## Research context

CLIP learns aligned image/text representations through contrastive image-text training (arXiv:2103.00020). Recent Visual-RAG and M2RAG benchmarks emphasize image evidence rather than text-only augmentation (arXiv:2502.16636, arXiv:2502.17297). ColPali/page-image retrieval is reserved for the later visual-document sub-lab.

## Definition of Done

- [x] raster benchmark with hidden visual state defined
- [x] text-surrogate BM25 control implemented
- [x] dependency-free pixel-native control implemented
- [x] multimodal fusion and no-evidence abstention implemented
- [x] retrieval, visual grounding, answer, and candidate-cost metrics separated
- [x] shared `Source` image provenance exposed
- [x] regression tests added, including complete frozen-raster validation
- [x] pinned pretrained CLIP retrieval control implemented on the same benchmark
- [ ] full repository CI + deterministic evaluator passes on the repaired frozen benchmark
- [ ] persisted deterministic results and failures reviewed
- [ ] pinned pretrained CLIP control completes successfully on the same benchmark
- [ ] CLIP result retained even if it underperforms the handcrafted control
- [ ] final findings/error analysis written down
- [ ] ROADMAP marks Multimodal RAG DONE only after final source-of-truth gate

This phase is a controlled mechanism study, not a claim that handcrafted color/layout features or a tiny synthetic CLIP benchmark generalize to production multimodal RAG.
