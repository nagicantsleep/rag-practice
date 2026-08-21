# M08.5 — Multimodal RAG

Status: **DONE** — final source-of-truth CI gate passed; ROADMAP completion follows this evidence.

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

The first candidate gate exposed malformed ASCII PPM payload lengths. Before any pretrained CLIP result was observed, all 9 benchmark rasters were regenerated from the already-declared visual semantics as explicit 8 rows × 8 RGB pixels. The frozen queries, qrels, captions, task labels, and intended marker/bar semantics were not changed after observing pretrained behavior. A regression test now parses every raster and asserts 64 RGB pixels.

## Evaluation contract

Retrieval is separated from answer quality:
- Recall@3 and Hit@1 on non-empty qrels;
- visual-required, cross-modal, and text-sufficient Hit@1;
- no-evidence accuracy;
- visual-evidence-grounded rate;
- answer correctness;
- visual candidates scored and query latency.

A short answer matching by accident does not repair a wrong retrieval trace.

## Results

Deterministic controls on the repaired frozen benchmark:

| System | Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Visual candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text surrogate BM25 | 0.875 | 0.500 | 0.333 | 0.500 | **1.000** | 0.000 | 0.200 | 0.000 | **0.0** |
| pixel-native | 0.625 | 0.500 | 0.667 | 0.000 | 0.000 | 0.000 | 0.500 | 0.667 | 7.2 |
| multimodal fusion | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 2.2 |

Pinned pretrained CLIP retrieval control:

| Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Visual candidates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.625 | 0.375 | 0.333 | 0.000 | 0.500 | 0.000 | 0.200 | 0.333 | 9.0 |

CLIP used 512-dimensional embeddings. The recorded CPU run loaded the model in about 6.16 s, embedded the 9-image corpus in about 516 ms, used 18,432 logical bytes for the image-vector matrix, and averaged about 26.1 ms per query. These timings are CI sanity measurements, not serving benchmarks.

## Findings and error analysis

- **Text evidence and visual evidence are not substitutes.** Text-surrogate BM25 reaches text-sufficient Hit@1 `1.0` but visual grounding `0.0`; it can retrieve a caption while still being unable to justify the requested visual state.
- **Pixel evidence without identity is insufficient for cross-modal questions.** The pixel-native control solves the four visual mechanisms well enough to reach visual Hit@1 `0.667`, yet cross-modal Hit@1 is `0.0` because `alpha`/`beta`, panel identity, site, and quarter are not pixel constraints.
- **The handcrafted fusion result is a mechanism control, not a production claim.** Its perfect scores come from explicit metadata constraints plus exact deterministic color/layout features on a tiny synthetic corpus. The point is evidence composition and failure separation, not general visual intelligence.
- **Pinned CLIP recovers some primitive visual semantics but does not solve the benchmark.** It ranks `p4` first for the red upper-left query and `p2` first for the green lower-right query, but misses the blue-vs-red chart at top 3 and ranks the yellow lower-left diagram only second.
- **CLIP does not recover external asset identity from pixels alone.** Both cross-modal queries fail at Hit@1 (`0.0`); for the beta/red query, `p5` is absent from top 3, and for alpha panel B/green, `p2` is only rank 2 behind `p3`. This is expected when captions/metadata are intentionally excluded from the image embedding path.
- **An embedding model is not an abstention policy.** Exhaustive CLIP always emits images, so both no-evidence cases fail (`0.0`). A production path needs explicit candidate filtering, score calibration/rejection, or evidence verification rather than assuming similarity implies evidence exists.
- **CLIP is not automatically cheaper or better on this controlled task.** It scores all 9 images per query and reaches Hit@1 `0.375`, while the explicit fusion scores 2.2 visual candidates on average and reaches Hit@1 `1.0`. This comparison is only valid for the frozen toy benchmark, not as a general model-ranking claim.
- **Answer correctness remains separate from retrieval quality.** CLIP answer correctness is `0.2`; the deterministic pixel control reaches `0.5` partly because a wrong retrieved image can accidentally share the requested short answer. Retrieval trace and modality provenance therefore remain first-class evaluation outputs.
- **The negative pretrained result is retained.** The benchmark was repaired and frozen before the first CLIP outcome, and the CLIP result is recorded without changing queries/qrels to make the pretrained model look better.

## Research context

CLIP learns aligned image/text representations through contrastive image-text training (arXiv:2103.00020). Recent Visual-RAG and M2RAG benchmarks emphasize image evidence rather than text-only augmentation (arXiv:2502.16636, arXiv:2502.17297). ColPali/page-image retrieval is reserved for the later visual-document sub-lab.

## Evaluation evidence

- Initial candidate gate exposed malformed PPM fixtures; no evaluator result from that failing gate was accepted.
- Successful repaired-benchmark PR gate `32460722561`: **110 tests passed**, deterministic multimodal evaluation passed, and the pinned CLIP evaluation passed.
- Push automation persisted deterministic and CLIP JSON/Markdown evidence in commit `9e299f6ba512022900de7273b388730f0ae51603`.
- Final source-of-truth gate `32460985448` passed on findings head `42a7b6e114f437e05d1adb6f984527ce04ee1dd8` with the full test suite, deterministic evaluator, and pinned CLIP evaluator all successful.

## Definition of Done

- [x] raster benchmark with hidden visual state defined
- [x] text-surrogate BM25 control implemented
- [x] dependency-free pixel-native control implemented
- [x] multimodal fusion and no-evidence abstention implemented
- [x] retrieval, visual grounding, answer, and candidate-cost metrics separated
- [x] shared `Source` image provenance exposed
- [x] regression tests added, including complete frozen-raster validation
- [x] pinned pretrained CLIP retrieval control implemented on the same benchmark
- [x] full repository CI + deterministic evaluator passes on the repaired frozen benchmark
- [x] persisted deterministic results and failures reviewed
- [x] pinned pretrained CLIP control completes successfully on the same benchmark
- [x] CLIP result retained even though it underperforms the handcrafted control
- [x] final findings/error analysis written down
- [x] final source-of-truth gate passes on the findings commit
- [x] ROADMAP marks Multimodal RAG DONE only after that gate

This phase is a controlled mechanism study, not a claim that handcrafted color/layout features or a tiny synthetic CLIP benchmark generalize to production multimodal RAG.
