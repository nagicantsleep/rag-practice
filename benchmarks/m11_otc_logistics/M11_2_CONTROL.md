# M11.2 integrated-copilot control contract

Status: **FROZEN BEFORE M11.2 IMPLEMENTATION**

This contract is frozen after M11.1 baseline evidence and before implementing the integrated copilot. It does **not** change the M11.0 benchmark semantics.

## Hypothesis

A bounded structured-first investigation loop with pre-exposure authorization/freshness/trust controls should improve evidence completeness and strict task success over the one-shot mixed baseline while eliminating stale and untrusted evidence exposure.

## Runtime source boundary

The runtime may read only the source files declared in `manifest.json` plus this implementation contract. It must never read `benchmark.json`, expected answers, qrels/evidence IDs, task classes, or forbidden-evidence labels.

## Fixed control order

Before evidence can enter the answer context:

1. **authorization** — deny a source family when the user role is not allowed;
2. **snapshot/version** — use the requested source snapshot and effective contract version at that snapshot time;
3. **trust** — explicitly untrusted documents are excluded before ranking/selection;
4. **relevance / relationship traversal** — only then select records/documents for the investigation.

A denied sensitive request must fail closed without reading the sensitive finance record.

## Bounded investigation loop

Maximum **4 source actions** per query. An action can return multiple records from one logically joined source operation.

Available actions:

- `order_context(order_or_customer)` — order, customer, shipment, current tracking history, invoice identifier and inventory state; finance payload is excluded;
- `finance_context(order_id)` — role-gated payment/credit state;
- `active_contract(customer_id)` — current trusted contract effective at the snapshot time;
- `policy_search(exception_code, detail)` — trusted SOP retrieval using the discovered exception/code and event text.

The planner is deterministic and qrel-blind. It may use question language, user role, and recorded observations. It may not use evaluator-only task labels.

## Planning rules

- Resolve an explicit order ID first when present; customer-name lookup is allowed for contract-only questions.
- Finance is called only when the question asks for payment/credit/finance state or a multi-hop investigation explicitly asks whether finance is blocking fulfillment.
- Contract is called only when SLA, commitment, contract, or breach semantics are requested.
- Policy search is called only after a structured observation yields a confirmed operational exception, or when an unknown-delay procedure is needed because the carrier explicitly reports delay without cause.
- No second policy call is allowed after a trusted SOP is selected.
- Stop early when the required requested fields are supportable, when authorization denies the request, when evidence is explicitly insufficient, or when authoritative sources conflict.

## Domain reasoning rules

These are source-schema rules, not benchmark labels:

- SLA elapsed time is measured from the earliest trusted `PICKED_UP` event to the snapshot `as_of` time and compared with the active contract commitment.
- A `DELAY_NOTICE` with no cause code yields `UNKNOWN` root cause; do not infer a cause.
- A trusted carrier `DELIVERED` event conflicting with current ERP shipment status yields `CONFLICT`.
- `credit_hold=true` is a finance blocker only when finance context is authorized and requested.
- `inventory.shortage=true` is an operational fulfillment blocker independent of finance.
- `PICKED_UP`, `LINEHAUL`, and `OUT_FOR_DELIVERY` are progress events, not exceptions.

## Evidence and trace contract

Persist per query:

- action sequence and action arguments;
- structured records returned by each action;
- documents considered and selected;
- rejected unauthorized/stale/untrusted counts and IDs;
- final evidence IDs;
- answer fields;
- stop reason;
- action count;
- stage and total latency.

## Comparison contract

Use the unchanged frozen M11 test split and evaluator. Compare against all M11.1 baselines on:

- strict task success;
- answer field accuracy;
- evidence recall and precision;
- source recall and precision;
- unauthorized/stale/untrusted exposure;
- conflict/no-evidence/mutation task success;
- action count and latency.

Do not tune the frozen benchmark, normalization, or expected labels after seeing M11.2 results. Any implementation defect repair must preserve this control contract and the original M11.0 benchmark semantics.
