# M11.2 — Integrated copilot findings

Status: **EVIDENCE RECORDED BEFORE M11.3 IMPLEMENTATION**

The M11.0 benchmark remains frozen at `96031c933f6b53b22fb50f0ca02e723a0d928aa1`. The M11.2 control contract in `benchmarks/m11_otc_logistics/M11_2_CONTROL.md` was frozen before integrated implementation and is unchanged.

## First integrated result

PR gate `32557702073` / job `96994409422` passed **157 repository tests**, the unchanged M11.1 baseline evaluator, and the first M11.2 integrated evaluator.

| System | Strict success | Field accuracy | Evidence recall | Evidence precision | Source recall | Source precision | Unauthorized exposure | Stale exposure | Untrusted exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed one-shot mixed baseline | 0.500 | 0.856 | 0.782 | 0.392 | 0.924 | — | 0.000 | 1.000 | 1.000 |
| first integrated copilot | 0.500 | 0.833 | 0.917 | 0.752 | 0.912 | 0.800 | 0.000 | 0.000 | 0.000 |

The first integrated control materially improved evidence discipline and removed stale/untrusted exposure, but did not initially improve strict task success. The failure inspection below was therefore completed before accepting the final M11.2 evidence.

## Control-preserving repair

The first result exposed implementation defects that violated intended frozen control semantics without requiring any benchmark, qrel, expected-answer, authorization, mutation, clock, or normalization change:

- natural requests such as “what should operations do next?” were not recognized as action intent, so a confirmed exception did not trigger the allowed targeted policy lookup;
- an authorized `credit_hold=true` finance observation was returned but not normalized to the domain blocker label `CREDIT_HOLD`;
- an inventory-only blocker did not explicitly record that finance was intentionally `NOT_READ` under the source-selection policy;
- customer-name resolution used for contract lookup was not included in provenance even though the frozen control explicitly allows that lookup;
- rejection of a retained untrusted candidate was traced but not reflected in the answer-level `ignored_untrusted` field, and a confirmed exception was not surfaced under the generic `exception` field;
- when escalation was requested before any confirmed operational exception, the system stopped safely but did not emit the explicit `NONE_YET` action state.

Commits `a93f6879c5fdb1dd1e047441ee1b50baa54e60b8` and `965a52f96cdc030eaba3ff2fcb7fc45f15d01c12` repair only those semantics and add contract-level regression tests. They do not read evaluator labels at runtime and do not change the frozen M11.0/M11.2 contracts.

## Final M11.2 evidence

Repaired PR gate `32558363822` / job `96996006860` passed **161 repository tests**, the unchanged M11.1 baseline evaluator, and the repaired M11.2 evaluator. The push workflow persisted `integrated_results.json` and `integrated_results.md` on the branch.

| System | Strict success | Field accuracy | Evidence recall | Evidence precision | Source recall | Source precision | Unauthorized exposure | Stale exposure | Untrusted exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed one-shot mixed baseline | 0.500 | 0.856 | 0.782 | 0.392 | 0.924 | — | 0.000 | 1.000 | 1.000 |
| repaired integrated copilot | **0.889** | **0.972** | **0.986** | **0.757** | **0.972** | **0.800** | **0.000** | **0.000** | **0.000** |

Additional final metrics:

- no-evidence task success: `1.0`;
- mutation task success: `1.0`;
- mean source actions: `1.778`;
- maximum source actions: `4`, exactly the frozen budget ceiling;
- rejected stale records/documents: `1`;
- rejected unauthorized source reads: `1`;
- rejected untrusted documents: `8`;
- GitHub Actions CPU mean latency is approximately `0.2–0.4 ms/query` across the evidence runs and is an implementation sanity number, not a production throughput claim.

## Retained failures

Two held-out failures remain intentionally unoptimized:

1. **Atlas contract-only task:** the expected answer includes `weather_waiver=false` although the question asks only for the delivery commitment. The system correctly retrieves `C-ATLAS` and `CTR-ATLAS-v1` and returns the requested 48-hour commitment, but does not synthesize an unrequested waiver field merely to satisfy the held-out label.
2. **Foxtrot conflict task:** the answer itself is fully correct (`CONFLICT`, ERP `IN_TRANSIT`, carrier `DELIVERED`), but the qrel also requires `CTR-FOX-v1`. The frozen M11.2 planner permits `active_contract` only for SLA/commitment/contract/breach intent, so the system stops on authoritative operational conflict rather than violating its frozen planner to recover that contract evidence.

These failures are retained as an important capstone lesson: strict benchmark success can expose a mismatch between evaluator evidence expectations and a pre-frozen runtime policy. M11.2 is not tuned after the fact to erase that mismatch.

Persisted evidence: `labs/11_otc_logistics/results/integrated_results.json` and `integrated_results.md`.
