# M03 Chunking Phase 1

Retriever is fixed to BM25; only the chunk/index representation changes.

| Strategy | Chunks | Mean words | Doc hit@1 | Evidence@1 | Evidence@3 | Source token util@3 | Relevant context@3 | Mean query ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_24 | 13 | 20.8 | 1.000 | 0.200 | 0.800 | 1.000 | 0.694 | 0.025 |
| fixed_24_overlap_8 | 17 | 21.6 | 1.000 | 0.400 | 1.000 | 0.886 | 0.752 | 0.022 |
| sentence_35 | 11 | 24.6 | 0.800 | 0.800 | 1.000 | 1.000 | 0.609 | 0.018 |
| paragraph_80 | 5 | 54.2 | 0.800 | 0.800 | 1.000 | 1.000 | 0.472 | 0.016 |
| semantic_50 | 19 | 14.3 | 0.800 | 0.200 | 0.800 | 1.000 | 0.764 | 0.020 |
| sentence_35_metadata | 11 | 39.5 | 1.000 | 0.800 | 1.000 | 0.635 | 0.742 | 0.021 |

`source_token_utilization@3` is the number of unique source-word positions represented in the retrieved context divided by the actual context word count. It therefore penalizes overlap and metadata prefix overhead instead of treating duplicated tokens as free.
