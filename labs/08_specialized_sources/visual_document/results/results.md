# M08.6 Visual-document / page-image RAG results

Benchmark: 6 frozen synthetic document pages, 10 queries (text, layout, table, chart, cross-modal, region, no-evidence).

| System | Recall@3 | Hit@1 | Visual Hit@1 | Cross-modal Hit@1 | Text Hit@1 | No-evidence | Answer correct | Visual grounded | Region locator | Visual candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ocr_surrogate | 0.875 | 0.750 | 0.667 | 1.000 | 1.000 | 0.500 | 0.400 | 0.000 | 0.000 | 0.0 |
| page_native | 0.750 | 0.750 | 1.000 | 1.000 | 0.000 | 1.000 | 0.700 | 1.000 | 1.000 | 4.2 |
| ocr_page_fusion | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.2 |

## Interpretation guardrails

- OCR surrogate retrieval only sees frozen text extraction; a page hit or correct short answer is not visual/layout evidence.
- Page-native retrieval sees exact raster pixels and layout markers but deliberately cannot read frozen OCR facts; it is a deterministic mechanism control, not a learned document model.
- OCR+page fusion keeps both modalities explicit and records page plus region locators rather than collapsing raster evidence into OCR text.
- No-evidence behavior is evaluated separately; visual similarity or lexical overlap is not automatically an abstention policy.
- The benchmark is tiny and synthetic. Perfect fusion scores demonstrate the controlled evidence mechanism only.
