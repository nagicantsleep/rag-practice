# M11.1 — Baseline findings

Status: **EVIDENCE RECORDED BEFORE M11.2 IMPLEMENTATION**

The M11.0 benchmark was frozen in commit `96031c933f6b53b22fb50f0ca02e723a0d928aa1`. Baseline implementation followed in `9ebdb7469b7179cd67696db811b5372be8d2131a`; no benchmark query, qrel/evidence label, expected answer, permission, snapshot, mutation, clock, or normalization was changed after first inspection.

Official PR gate `32518642131` / job `96885807414` passed **153 repository tests** and the frozen M11.1 evaluator. Push evidence was persisted in bot commit `b494c8bf59ceee79b1503bbc4747d02650dcba3a`.

| System | Strict task success | Field accuracy | Evidence recall | Evidence precision | Source recall | Unauthorized exposure | Stale exposure | Untrusted exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no retrieval | 0.000 | 0.046 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| document-only BM25 | 0.000 | 0.167 | 0.156 | 0.148 | 0.244 | 0.000 | **1.000** | **1.000** |
| structured-only | 0.333 | 0.537 | 0.626 | 0.619 | 0.680 | 0.000 | 0.000 | 0.000 |
| fixed one-shot mixed-source | **0.500** | **0.856** | **0.782** | 0.392 | **0.924** | 0.000 | **1.000** | **1.000** |

Strict task success requires every expected structured answer field, every required evidence ID, and zero forbidden evidence exposure.

## Findings

- **One-shot source breadth is not evidence discipline.** The fixed mixed baseline reaches field accuracy `0.856` and source recall `0.924`, but its low evidence precision and perfect stale/untrusted exposure rates show that touching more sources does not make the investigation safe or complete.
- **Security and retrieval quality are independent.** Structured-only and fixed-mixed correctly deny unauthorized finance access before reading the sensitive finance record, while naive document retrieval still exposes stale and untrusted documents on other tasks.
- **Freshness must be a pre-retrieval/evidence rule.** Epsilon's expired `CTR-EPS-v1` remains retrievable alongside current `CTR-EPS-v2`; naive BM25 returns both, so a downstream answer can be numerically correct while the evidence contract still fails.
- **Trust must be enforced before evidence exposure.** Gamma's explicitly untrusted prompt-injection note is retrieved by naive document search. The benchmark treats merely surfacing that document as a failure even if the final action happens to be correct.
- **Structured-only retrieval cannot finish policy work.** It can join order/shipment/invoice/inventory/finance state, but cannot supply contract commitments or exception-specific SOP evidence.
- **The one-shot mixed baseline exposes the need for an actual retrieval loop.** It searches documents only with the original question; after structured retrieval discovers `WX_HOLD`, `CUSTOMS_REVIEW`, or `VEHICLE_BREAKDOWN`, it cannot issue a targeted second document query for the corresponding SOP.
- **Unknown and conflict are first-class outcomes.** Delta has a delay notice without confirmed root cause; Foxtrot has ERP/carrier disagreement. The integrated system must stop with explicit uncertainty/conflict rather than force a single explanation.

These results are mechanism controls on a small deterministic synthetic benchmark, not production-quality or throughput claims.
