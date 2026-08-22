# M11.2 — Integrated copilot findings

Status: **INITIAL EVIDENCE RECORDED / CONTROL-PRESERVING REPAIR UNDER EVALUATION**

The M11.0 benchmark remains frozen at `96031c933f6b53b22fb50f0ca02e723a0d928aa1`. The M11.2 control contract in `benchmarks/m11_otc_logistics/M11_2_CONTROL.md` was frozen before integrated implementation and is unchanged.

## First integrated result

PR gate `32557702073` / job `96994409422` passed **157 repository tests**, the unchanged M11.1 baseline evaluator, and the first M11.2 integrated evaluator.

| System | Strict success | Field accuracy | Evidence recall | Evidence precision | Source recall | Source precision | Unauthorized exposure | Stale exposure | Untrusted exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed one-shot mixed baseline | 0.500 | **0.856** | 0.782 | 0.392 | **0.924** | — | 0.000 | 1.000 | 1.000 |
| first integrated copilot | 0.500 | 0.833 | **0.917** | **0.752** | 0.912 | **0.800** | 0.000 | **0.000** | **0.000** |

The first integrated control therefore materially improved evidence discipline and removed stale/untrusted exposure, but did **not** improve strict task success over the fixed mixed baseline. The hypothesis is only partially supported at this point.

Additional first-result metrics:

- mean source actions: `1.667`;
- maximum source actions: `4` (within the frozen budget);
- conflict task success: `0.0`;
- no-evidence task success: `0.0`;
- mutation task success: `0.5`;
- rejected stale records/documents: `1`;
- rejected unauthorized source reads: `1`;
- rejected untrusted documents: `8`;
- mean GitHub Actions CPU runtime: about `0.192 ms/query` for this deterministic educational implementation.

## Failure inspection and repair boundary

The first result exposed several implementation defects that violate the intended frozen control semantics without requiring any benchmark, qrel, expected-answer, authorization, mutation, clock, or normalization change:

- natural requests such as “what should operations do next?” were not recognized as action intent, so a confirmed exception did not trigger the allowed targeted policy lookup;
- an authorized `credit_hold=true` finance observation was returned but not normalized to the domain blocker label `CREDIT_HOLD`;
- an inventory-only blocker did not explicitly record that finance was intentionally `NOT_READ` under the source-selection policy;
- customer-name resolution used for contract lookup was not included in provenance even though the frozen control explicitly allows that lookup;
- rejection of a retained untrusted candidate was traced but not reflected in the answer-level `ignored_untrusted` field, and a confirmed exception was not surfaced under the generic `exception` field;
- when escalation was requested before any confirmed operational exception, the system stopped safely but did not emit the explicit `NONE_YET` action state.

Commits `a93f6879c5fdb1dd1e047441ee1b50baa54e60b8` and `965a52f96cdc030eaba3ff2fcb7fc45f15d01c12` repair only those semantics and add contract-level regression tests. They do not read evaluator labels at runtime and do not change the frozen M11.0/M11.2 contracts.

Two first-result failures are intentionally **not** repaired from held-out labels:

1. the Atlas contract-only task requires `weather_waiver=false` although the question asks only for the delivery commitment; the integrated control does not synthesize unrequested fields solely to satisfy that held-out label;
2. the Foxtrot conflict task requires contract evidence even though the M11.2 planning contract permits `active_contract` only for SLA/commitment/contract/breach intent. The system stops after authoritative operational conflict rather than violating its frozen planner to satisfy the qrel.

The next accepted evidence is the repaired full gate on the unchanged frozen benchmark. If it succeeds, remaining failures are retained as control/benchmark trade-offs rather than tuned away.
