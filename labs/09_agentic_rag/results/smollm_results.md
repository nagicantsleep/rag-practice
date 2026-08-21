# M09 pinned SmolLM2 tool-planner results

Model: `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`

| Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Tool cost | Planner calls | Valid decisions | Prompt tokens | Planner generation ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.167 | 0.167 | 0.000 | 0.000 | 0.167 | 1.000 | 0.000 | 0.00 | 0.00 | 1.00 | 0.000 | 153.2 | 326.1 |

The model selects tools only. Final answers are produced by the same qrel-blind deterministic evidence reader used by phase 1, isolating planner/tool-selection behavior.
Raw planner outputs are persisted without expected-answer-aware repair or post-hoc action cleanup.
