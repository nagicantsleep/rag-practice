# M08.5 Multimodal RAG results

Benchmark: 9 raster images, 10 queries (visual-only, text-sufficient, cross-modal, no-evidence).

| System | Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Visual candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| text_surrogate | 0.875 | 0.500 | 0.333 | 0.500 | 1.000 | 0.000 | 0.200 | 0.000 | 0.0 |
| pixel_native | 0.625 | 0.500 | 0.667 | 0.000 | 0.000 | 0.000 | 0.500 | 0.667 | 7.2 |
| multimodal_fusion | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.2 |

## Interpretation guardrails

- Text-surrogate retrieval never inspects pixels; a caption hit is not visual evidence.
- Pixel-native retrieval deliberately ignores asset/site metadata, so it can solve visual-only questions yet fail cross-modal identity constraints.
- Multimodal fusion applies explicit site/kind constraints, then combines BM25 surrogate relevance with pixel evidence when the query asks for a visual property.
- Answer correctness is reported separately from retrieval: a wrong image can accidentally yield the same short answer string.
- The raster parser and visual features are deterministic teaching controls, not a learned vision model.
