# M08.5 pretrained CLIP retrieval control

Model: `openai/clip-vit-base-patch32` @ `b97b0100e55e367c057773c2a614676470b0d575`

The CLIP control receives only query text and image pixels for ranking. Titles, captions, site metadata, qrels, and answer labels are not exposed to the retriever.

| Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Visual candidates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.625 | 0.375 | 0.333 | 0.000 | 0.500 | 0.000 | 0.200 | 0.333 | 9.0 |

## Interpretation guardrails

- This is an exhaustive text-to-image CLIP retrieval control, not a multimodal fusion system.
- Image embeddings are built from the frozen P3 raster pixels only; captions and metadata are excluded.
- The existing deterministic pixel reader is used only after retrieval to score answer correctness separately from retrieval quality.
- CLIP always returns ranked images here, so no-evidence accuracy tests whether a retrieval-only control can abstain without an explicit rejection mechanism.
- The benchmark is tiny and synthetic; the result is retained whether it helps or hurts relative to the handcrafted controls.
