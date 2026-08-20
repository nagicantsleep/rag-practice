# M03 Parent-Child and Hierarchical Retrieval

BM25 scoring is retained at every searchable layer so this phase isolates representation/routing choices.

| Strategy | Returned context | Doc hit@1 | Evidence@1 | Evidence@3 | Source token util@3 | Relevant context@3 | Searchable index words | Route hit@1 | Mean query ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_24_overlap_8 | chunk | 1.000 | 0.400 | 1.000 | 0.886 | 0.752 | 367 | — | 0.036 |
| sentence_35_metadata | chunk | 1.000 | 0.800 | 1.000 | 0.635 | 0.742 | 434 | — | 0.028 |
| parent_child | paragraph parent | 0.800 | 0.600 | 1.000 | 1.000 | 0.533 | 271 | — | 0.034 |
| hierarchical_metadata_root | plain sentence leaf | 1.000 | 1.000 | 1.000 | 1.000 | 0.763 | 601 | 1.000 | 0.049 |

Parent-child indexes narrow children but returns wider parents. Hierarchical routing stores a document-level metadata+body root representation and returns plain sentence leaves, so metadata can affect routing without consuming answer-context tokens.
