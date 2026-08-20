# M07 structured retrieval results

## Retrieval / graph / hierarchy

| System | Recall@3 | Recall@5 | Evidence complete@budget | MRR | Mean query ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat_bm25 | 0.447 | 0.567 | 0.100 | 0.660 | 0.027 |
| flat_metadata_bm25 | 0.497 | 0.700 | 0.100 | 0.708 | 0.025 |
| raptor_style | 0.630 | 0.683 | 0.300 | 0.725 | 0.151 |
| kag_path | 0.717 | 0.717 | 0.600 | 0.800 | 0.024 |
| graph_global | 0.580 | 0.767 | 0.600 | 0.808 | 0.066 |
| light_rag_dual | 0.880 | 0.933 | 1.000 | 1.000 | 0.047 |
| hipporag_ppr | 0.630 | 0.767 | 0.200 | 0.808 | 0.286 |

## Memory / updates

| System | Hit@1 | Current Hit@1 | Previous Hit@1 | Stale current rate | Mean query ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat_bm25_all_versions | 0.250 | 0.000 | 1.000 | 1.000 | 0.019 |
| temporal_memory | 1.000 | 1.000 | 1.000 | 0.000 | 0.025 |
