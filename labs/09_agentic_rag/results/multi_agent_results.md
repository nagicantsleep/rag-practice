# M09 exploratory role-split multi-agent results

Shared model: `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f77366fa6b3b4b768ec3050bf629380bac`

This is a post-single-agent exploratory control on the same frozen benchmark, not fresh held-out generalization evidence.

| Task success | Grounded | Plan exact | Tool precision | Evidence complete | Abstention | Recovery | Steps | Tool cost | Role calls | Proposer valid | Critic valid | Proposer ms | Critic ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.167 | 0.167 | 0.000 | 0.000 | 0.167 | 1.000 | 0.000 | 0.00 | 0.00 | 2.00 | 0.000 | 0.000 | 333.3 | 819.2 |

The proposer is unchanged from the recorded single-agent control. The critic/corrector is a new role sharing the same pinned weights; both raw outputs are persisted.
No evaluator labels are available to either role or the coordinator.
