# M08.7 — Long-context vs retrieval routing

Status: **IN PROGRESS — FROZEN BENCHMARK / MECHANISM GATE PENDING**

## Goal

Measure when direct full-context reading, retrieval-first context selection, or an explicit router should be preferred while keeping route quality, evidence completeness, answer correctness, grounding, abstention, latency, retrieval calls, and context footprint separate.

## Frozen systems

1. `always_direct` — gives the reader every section in the query's frozen context bundle.
2. `always_retrieve` — BM25 over section text with the frozen top-2 budget.
3. `explicit_router` — reads the whole bundle only when it is small (<=100 tokenizer words) or the query contains a frozen global marker; otherwise it retrieves.
4. A pinned pretrained reader control will be added only after the deterministic benchmark/mechanism gate is stable.

The deterministic reader is a qrel-blind mechanism control. It derives supported benchmark answers from the selected text and returns `ABSTAIN` when the requested evidence pattern is absent.

## Definition of Done

- [x] Freeze bundle templates/text, section boundaries, queries, qrels, expected answers, preferred routes, retrieval depth, size threshold, and global markers before pretrained inspection.
- [ ] Implement direct, retrieval-first, and explicit routing controls.
- [ ] Evaluate route accuracy, evidence recall/completeness, answer correctness, grounding, abstention, context footprint, retrieval calls, and latency separately.
- [ ] Pass full-regression + deterministic evaluation CI.
- [ ] Add a pinned pretrained reader on the unchanged frozen benchmark and retain failures.
- [ ] Persist deterministic + pretrained JSON/Markdown results.
- [ ] Record error analysis and final source-of-truth gate.
- [ ] Mark M08.7 and M08 complete in ROADMAP.

## Guardrails

The frozen benchmark is tiny and synthetic. Perfect route decisions on it are a mechanism demonstration, not a learned router or production long-context claim. The preferred route encodes the declared cost/evidence contract and is never available to runtime routing.
