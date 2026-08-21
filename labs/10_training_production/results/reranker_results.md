# M10.1b learned reranker results

| Split | Candidate Recall@3 | Rerank Recall@1 | MRR | Mean rerank margin | Mean rerank ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 1.000 | 0.875 | 0.938 | 3.9366 | 0.0480 |
| test | 1.000 | 1.000 | 1.000 | 5.5844 | 0.0453 |

Training pairs: `16`; training ms: `32.6`.

This is a transparent post-hoc training control on the unchanged M10.1 split. Candidate recall is measured before reranking; missing positives cannot be repaired by the reranker.
