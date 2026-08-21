# M08.7 frozen long-context vs retrieval benchmark

This benchmark is frozen before any pretrained reader result is inspected.

## Hypothesis

A single policy should not send every question through retrieval or every question through full-context reading.

- Small context bundles can be read directly without paying retrieval overhead.
- Sparse facts in larger bundles should benefit from retrieval because a tiny evidence window is enough.
- Global/count/list questions can require evidence from more sections than the frozen retrieval budget can return, so direct full-context reading can be safer.
- No-evidence behavior must stay explicit; retrieving a similar section is not proof that an answer exists.

The benchmark is a controlled routing/evidence exercise, not a claim about production long-context transformers.

## Corpus

Four frozen context bundles contain natural-language sections:

- `cedar_brief`: short context, 46 tokenizer words.
- `atlas_handbook`: long repeated-process context, 779 tokenizer words.
- `orion_report`: long repeated-process context, 650 tokenizer words.
- `lumen_notes`: medium context, 341 tokenizer words.

The repeated neutral process language is deliberate. It creates lexical distractors while keeping the evidence statements human-auditable.

## Queries

Twelve frozen queries cover short direct-reading facts, long sparse facts, two-section comparison, global count/list questions, and short/long no-evidence cases.

Each query records section qrels, an expected answer for evaluation only, and a frozen `preferred_route` (`direct` or `retrieve`).

## Frozen routing contract

- BM25 retrieval budget: top `2` sections.
- Direct-size threshold: `100` tokenizer words.
- Explicit router chooses `direct` when the bundle is at or below the size threshold, or when the query contains a frozen global marker (`across the entire`, `list every`); otherwise it chooses `retrieve`.
- `ABSTAIN` is the only no-evidence answer.

The router may inspect only the question and bundle size. It must not inspect qrels, expected answers, answerability labels, or pretrained reader outputs.

## Integrity rules

- Do not change templates, bundle sections/special evidence, queries, qrels, expected answers, preferred routes, retrieval depth, direct-size threshold, or global markers after observing pretrained reader behavior.
- Evaluate route correctness separately from answer correctness.
- Evaluate evidence completeness separately from answer correctness and grounding.
- Record context words / context fraction, retrieval calls, latency, and unnecessary retrieval/full-context use.
- A deterministic reader is a mechanism control. A pretrained reader, if added, runs on this unchanged benchmark with a pinned model revision and its negative cases retained.
