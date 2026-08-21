# M11 benchmark construction contract

Status: **CONSTRUCTION CONTRACT FROZEN — INSTANCES NOT YET FROZEN**

This file governs how the M11 Order-to-Cash & Logistics Exception Resolution Copilot benchmark is built. The actual benchmark examples, qrels, expected answers, role permissions, source versions, and mutation timeline will be frozen in a later commit **before** any optimized M11 retrieval, prompt, routing, or agent implementation is inspected.

## Data model requirements

The versioned corpus must contain linked entities from at least:

- customers;
- sales orders and order lines;
- inventory / fulfillment state;
- shipments;
- tracking events / carrier exceptions;
- invoices;
- payment status and credit-hold state;
- customer contracts / SLA documents;
- operational SOPs / escalation policies;
- user / role authorization state.

Every structured record and document must have a stable identifier. Mutable records / documents must expose a source version or event timestamp so the evaluator can distinguish current from stale evidence.

## Relationship requirements

The benchmark must include cross-source joins that cannot be solved reliably by one flat text search, including:

```text
customer_id → order_id → shipment_id → tracking events
order_id → invoice_id → payment / credit state
customer_id → contract_id / SLA version
exception_code / exception class → SOP / escalation policy
```

## Benchmark split policy

Use a held-out split that prevents trivial memorization of exact operational cases. Entity families used in held-out evaluation should not simply duplicate training cases under new IDs.

The freeze commit must record:

- random / deterministic seed if synthetic generation is used;
- split policy and entity allocation;
- source versions and benchmark clock;
- all mutations that occur during temporal scenarios;
- qrels / evidence IDs;
- expected answers or structured expected fields;
- authorization context per query;
- no-evidence and conflict labels;
- evaluator normalization rules;
- tool / route labels only where needed for control metrics.

## Required task classes

The frozen benchmark must contain enough examples to evaluate each class separately:

1. simple structured lookup;
2. policy / contract document lookup;
3. structured join;
4. cross-source document + structured composition;
5. multi-hop delay / exception investigation;
6. SLA breach / non-breach decision;
7. finance or fulfillment blocker attribution;
8. temporal current-state query;
9. stale-evidence rejection;
10. unauthorized query that must fail closed;
11. authorized version of a sensitive query;
12. no-evidence / unknown root cause;
13. conflicting-source evidence;
14. adversarial or prompt-injection evidence;
15. mutation case where the correct answer changes after an update.

## Baseline fairness requirements

All systems compared on the same task must see the same underlying corpus snapshot and authorization context. A method must not receive evaluator labels unavailable to another method.

When a source family is intentionally unavailable to a baseline, report that limitation explicitly rather than silently giving it substitute evidence.

## Integrity rule

After the actual benchmark freeze commit:

- do not change held-out queries, qrels, expected answers, authorization context, source versions, mutation sequence, benchmark clock, or evaluation normalization in response to model outputs;
- objective serialization, compatibility, or implementation defects may be repaired only if semantic benchmark evidence is unchanged and the repair is documented;
- prompt, retrieval, routing, reranking, agent, cache, or generation changes must be evaluated against the same frozen benchmark;
- representative failures are retained rather than removed to improve aggregate metrics.

## Minimum persisted result schema

Every evaluated system must persist enough per-query data to reconstruct why it succeeded or failed:

- query / task ID and class;
- user role / authorization context;
- source snapshot / generation;
- selected route and tool actions;
- structured records read;
- retrieved document IDs / scores;
- rejected stale / unauthorized / untrusted evidence counts;
- final evidence IDs;
- answer / structured decision;
- citations / provenance;
- abstention / conflict decision;
- latency and cost fields;
- evaluator metrics.

## Next freeze boundary

The next evidence boundary is the commit that adds the complete benchmark instances and marks this directory **FROZEN BEFORE M11 SYSTEM IMPLEMENTATION**. No optimized M11 pipeline should be implemented before that commit.
