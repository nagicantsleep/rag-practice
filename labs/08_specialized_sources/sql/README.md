# M08.2 — SQL / Structured RAG

Status: **IN PROGRESS** — implementation/evaluation candidate pending CI evidence.

## Hypothesis

Flattening database rows into text loses the operations that make structured sources useful: joins, filters, aggregation, grouping, empty-result semantics, and read-only safety. A structured RAG path should therefore evaluate SQL planning/execution separately from row-level provenance and final answer formatting.

## Mechanism

Shared boundary from M08.1:

`Source.search(query, limit) -> SourceHit[SourceRecord]`

Structured extension:

`schema discovery -> transparent query plan -> read-only validation -> SQL execution -> row-level evidence query -> deterministic answer + sqlite:// citations`

`SQLiteStructuredSource.search()` remains a flat row-BM25 control implementing the shared `Source` protocol. Structured execution is explicit rather than hidden behind the search contract.

The database runs with `PRAGMA query_only=ON`; the validator accepts only `SELECT`/`WITH` and rejects mutating/administrative SQL before execution.

## Controlled benchmark

`benchmarks/m08_sql/` defines a four-table commerce database (`customers`, `orders`, `order_items`, `products`) and nine queries covering:

- direct lookup;
- join retrieval;
- aggregate revenue;
- aggregate + join ranking;
- group-by;
- empty result;
- filtered aggregate;
- unsafe mutation;
- unsupported schema/question.

The planner is deliberately rule-based and inspectable. It receives the question and discovered schema, never qrels/reference answers.

## Baseline

Flat BM25 over serialized database rows. This is intentionally a retrieval-only control: it can retrieve row-shaped evidence but cannot execute relational operators.

## Evaluation contract

Structured execution:
- safe-query execution success;
- answer exact match;
- empty-result accuracy;
- unsafe rejection rate;
- unsupported-question fail-closed rate.

Evidence/provenance:
- row-level evidence recall;
- exact evidence completeness;
- schema table precision/recall;
- `sqlite://...` citations.

System behavior:
- planning, SQL, evidence, and end-to-end latency;
- explicit planned tables and SQL trace.

Answer correctness and evidence completeness are separate: an aggregate may produce the right scalar while omitting contributing rows, which must still count as a provenance failure.

## Definition of Done

- [x] shared Source boundary reused for a flat-row control
- [x] SQLite schema discovery and row locators implemented
- [x] transparent query planner implemented
- [x] read-only validation and query-only execution implemented
- [x] join/aggregate/group-by/empty/unsafe benchmark defined
- [x] row-level evidence citations implemented
- [x] source/execution/evidence metrics separated
- [x] regression tests added
- [ ] full repository CI + SQL evaluator passes
- [ ] persisted JSON/Markdown results reviewed
- [ ] representative failure cases written down
- [ ] ROADMAP marks SQL / Structured RAG DONE only after evidence passes

Research context: Spider (arXiv:1809.08887) popularized cross-domain text-to-SQL evaluation, while BIRD (arXiv:2305.03111) emphasizes larger databases and realistic schema/value reasoning. This lab is intentionally much smaller and mechanism-focused; it does not claim general text-to-SQL ability.
