# M09 agentic RAG deterministic results

| System | Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Cost units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| docs_only | 0.250 | 0.250 | 0.167 | 0.750 | 0.250 | 1.000 | 0.000 | 1.00 | 2.00 |
| static_router | 0.417 | 0.417 | 0.333 | 1.000 | 0.417 | 1.000 | 0.000 | 1.00 | 1.54 |
| agent_loop | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.92 | 2.58 |

Latency is a local/GitHub Actions CPU sanity measurement; tool costs are the frozen synthetic cost units.
Per-task JSON retains action arguments, observations, evidence ids, recovery count, latency, and cost.
