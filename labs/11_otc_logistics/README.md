# M11 — Order-to-Cash & Logistics Exception Resolution Copilot

Status: **CHARTER FROZEN — BENCHMARK CONSTRUCTION NEXT**

M11 is the real-world capstone after M00–M10. It integrates structured ERP data, logistics events, operational documents, policy controls, retrieval/reranking, agentic multi-step investigation, freshness, permissions, caching, observability, and explicit abstention into one production-oriented system.

## Product objective

Build an evidence-grounded assistant that investigates order and shipment exceptions across ERP, logistics, finance, contracts, and SOP sources, then returns:

1. root cause or the strongest evidence-supported explanation;
2. SLA / contractual impact;
3. order, shipment, invoice, inventory, and payment status when authorized;
4. recommended operational next action;
5. exact source citations / record provenance;
6. explicit uncertainty or abstention when evidence is missing or conflicting;
7. an auditable retrieval/tool trace.

Example task:

> Order SO-10482 missed its ETA. Determine why it is delayed, whether the customer SLA is breached, whether finance is blocking fulfillment, and which escalation procedure applies.

## Frozen source families

M11 must combine at least these source families rather than reducing the capstone to document-only RAG:

- **ERP / order management** — sales orders, customers, line items, inventory, fulfillment state;
- **finance** — invoices, payment status, credit-hold state;
- **logistics / TMS** — shipments, carrier, tracking events, ETA, exception codes;
- **contracts / SLA** — customer service commitments, delivery windows, penalties, Incoterms where relevant;
- **operations knowledge** — delivery exception SOPs, escalation matrix, refund / notification policies;
- **authorization state** — user identity / roles that constrain record and document visibility.

Synthetic or openly publishable data may be used, but relationships and failure modes must resemble real operational systems. No confidential production data is required.

## Frozen investigation shape

The capstone must support both direct and multi-hop paths, for example:

```text
customer → sales_order → shipment → tracking_event
                     └→ invoice / payment / credit_hold
customer → contract / SLA
shipment exception → SOP / escalation policy
```

The runtime may use deterministic routing, SQL / structured lookup, lexical or dense retrieval, reranking, and bounded agentic tool loops, but benchmark labels, expected answers, qrels, and evaluator-only annotations must never be visible to the runtime.

## Phase boundaries

### M11.0 — Dataset and benchmark

Construct a realistic, versioned O2C/logistics dataset and freeze the actual train/dev/test or dev/test benchmark **before** optimizing retrieval, routing, prompts, or agents.

Required benchmark task classes:

- simple structured lookup;
- document / policy lookup;
- structured join;
- cross-source evidence composition;
- multi-hop exception investigation;
- SLA breach determination;
- finance / fulfillment blocker attribution;
- temporal / current-state questions;
- stale-document rejection;
- role / permission denial and allowed access;
- no-evidence / unknown root cause;
- conflicting evidence;
- adversarial / prompt-injection document;
- mutation / freshness case where an update changes the correct answer.

### M11.1 — Baseline system

Establish the simplest runnable baselines before improvements, including at minimum:

1. document-only RAG baseline;
2. structured-only lookup baseline;
3. fixed one-shot mixed-source pipeline;
4. no-retrieval / abstention reference where applicable.

### M11.2 — Integrated copilot

Build the bounded production-oriented investigation pipeline:

```text
auth / policy
  → route / plan
  → structured + document retrieval
  → rerank / evidence selection
  → evidence sufficiency check
  → optional bounded retry / next tool
  → answer + exact citations
  → audit trace
```

### M11.3 — Production gate

Add incremental ingestion, cache invalidation, source/version freshness, ACL enforcement before evidence exposure, telemetry, regression evaluation, and deterministic load / scale sanity checks.

## Frozen evaluation axes

Do not collapse these into one aggregate score.

### Retrieval / evidence

- structured record correctness;
- document Recall@K / MRR where applicable;
- evidence completeness;
- source / citation precision and recall;
- stale-evidence usage rate;
- unauthorized-evidence exposure rate;
- adversarial / untrusted-evidence exposure rate.

### Investigation / answer

- root-cause correctness;
- SLA-impact correctness;
- financial-status correctness when authorized;
- recommended-action correctness;
- groundedness / evidence support;
- abstention accuracy on no-evidence tasks;
- conflict-handling correctness.

### Agent / control

- route / tool precision and recall;
- exact or acceptable action sequence where defined;
- recovery success after a failed / empty lookup;
- stop correctness;
- unnecessary tool calls / actions;
- bounded-loop compliance.

### Production

- p50 / p95 latency by stage and end-to-end;
- token / model / synthetic tool cost;
- cache hit rate and invalidation correctness;
- freshness correctness after source mutation;
- observability completeness;
- index / storage footprint;
- deterministic scale / concurrency sanity measurements.

## Release philosophy

M11 is not complete because a demo looks convincing. A release candidate must pass a frozen regression gate. Retrieval quality, answer quality, security / authorization, freshness, and system cost are independent contracts: improvement in one must not hide regression in another.

## Definition of Done

- [x] Capstone use case and product objective frozen before implementation.
- [x] Required source families and task classes frozen before implementation.
- [x] Evaluation axes and phase boundaries frozen before implementation.
- [ ] Construct versioned realistic ERP / finance / logistics / contract / SOP dataset.
- [ ] Freeze benchmark instances, qrels, expected answers, permissions, source versions, mutations, and evaluation rules before model/prompt optimization.
- [ ] Implement and evaluate explicit baselines.
- [ ] Implement integrated mixed-source copilot with bounded planning / retrieval loop.
- [ ] Enforce ACL, freshness, trust, and no-evidence behavior before answer generation.
- [ ] Persist exact citations, records, tool actions, observations, and per-stage traces.
- [ ] Add incremental ingestion, mutation-aware cache invalidation, and regression tests.
- [ ] Measure quality, security, freshness, latency, cost, and scale separately.
- [ ] Inspect and retain representative failures instead of tuning them away post-hoc.
- [ ] Pass final source-of-truth CI gate and document findings / trade-offs.

## Immediate next step

Build **M11.0 dataset + benchmark only**. Do not implement the optimized retrieval / agent pipeline until the benchmark instances and integrity rules have been frozen in a separate commit.
