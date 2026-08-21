# M08.7 pinned SmolLM2 long-context routing results

Reader: `HuggingFaceTB/SmolLM2-135M-Instruct` pinned to `12fd25f77366fa6b3b4b768ec3050bf629380bac`; CPU/float32 greedy generation.

The reader receives only the question plus context chosen by each route policy. Qrels, expected answers, preferred routes, and answerability labels are excluded from the prompt.

| System | Route acc | Evidence complete | Answer acc | Grounded | Abstention | Context words | Prompt tokens | Retrieval calls | Generation ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smollm_always_direct | 0.583 | 1.000 | 0.000 | 0.167 | 0.000 | 490.5 | 725.4 | 0.00 | 2458.50 |
| smollm_always_retrieve | 0.417 | 0.700 | 0.000 | 0.000 | 0.000 | 100.2 | 216.1 | 1.00 | 1202.78 |
| smollm_explicit_router | 1.000 | 1.000 | 0.000 | 0.083 | 0.000 | 275.5 | 446.2 | 0.42 | 1666.07 |

## Runtime / representation

- model load: 4016.51 ms
- parameters: 134515008 / 538060032 logical bytes
- tokenizer model max length: 8192
- torch: `2.11.0+cu130`; transformers: `5.15.1`

## Guardrails

- This is one tiny pinned reader on a frozen synthetic benchmark, not a general long-context leaderboard claim.
- Retrieval evidence completeness and reader answer quality are recorded separately; a model failure is not credited to routing, and a retrieval miss is not hidden by generation.
- Raw generated answers are scored as emitted. No expected-answer-aware cleanup or post-hoc extraction is added.
- The explicit router remains deterministic and qrel-blind; pretrained outputs do not influence its decisions.
