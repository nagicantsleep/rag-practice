# M08.2 — SQL / Structured RAG

Status: **DONE** — final full-suite + SQL RAG gate passed in GitHub Actions run `32449677217`.

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

`benchmarks/m08_sql/` defines a four-table, 20-row commerce database (`customers`, `orders`, `order_items`, `products`) and nine queries covering:

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

## Persisted evaluation

Initial PR gate `32449251416` passed the full repository suite (**89 tests**) and the SQL / Structured RAG evaluator.

Final source-of-truth gate `32449677217` passed the same full-suite/evaluator sequence before this completion update.

| System | Evidence recall | Evidence complete | Answer exact | Execution success | Unsafe reject | Empty correct | Unsupported handled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat row BM25@5 | 0.500 | 0.500 | n/a | n/a | n/a | n/a | n/a |
| schema-aware validated SQL | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

The structured path also reached schema table precision/recall `1.000/1.000`. Mean SQLite execution latency was about `0.051 ms` and mean end-to-end control-path latency about `0.531 ms` on this 20-row in-memory benchmark.

## Error analysis / findings

- **Flat row retrieval is not relational execution.** `s3` asks for total shipped revenue; BM25 retrieves order rows but none of the five contributing `order_items`, so row-level evidence completeness is zero even though the retrieved rows look lexically relevant.
- **Joins and aggregates expose provenance gaps.** On `s4` (highest-revenue active enterprise customer) and `s7` (North-region revenue), flat BM25 again returns customers/orders while missing the item rows that actually determine the aggregate.
- **A lexical baseline can retrieve useful rows without being able to answer.** On `s2`, BM25 retrieves both relevant `order_items`, but the requested product names still require joining to `products`; retrieval success alone is not execution correctness.
- **Empty is a valid answer state.** `s6` returns `NO_ROWS` with no fabricated citations.
- **Unsafe mutation fails before execution.** `s8` plans a `DELETE`, the validator rejects it, and regression tests verify the order count is unchanged.
- **Unsupported schema intent fails closed.** `s9` requests a loyalty-tier concept absent from the planner/schema path and returns `planning_error` rather than inventing a column.
- **The perfect structured score is controlled-mechanism evidence, not general text-to-SQL.** The rule-based planner was written for this frozen schema/query family; Spider/BIRD-style cross-domain generalization is explicitly out of scope.
- **The SQL guardrail is educational defense-in-depth, not a full production sandbox.** It combines explicit statement validation with SQLite `PRAGMA query_only=ON`; real systems still need database permissions, query budgets, tenant filters, and a robust SQL parser/policy layer.
- **Latency numbers are sanity checks only.** The database is in-memory and tiny, so they are not serving or warehouse-scan claims.

Machine-readable evidence: `results/results.json`. Human-readable aggregate table: `results/results.md`.

## Definition of Done

- [x] shared Source boundary reused for a flat-row control
- [x] SQLite schema discovery and row locators implemented
- [x] transparent query planner implemented
- [x] read-only validation and query-only execution implemented
- [x] join/aggregate/group-by/empty/unsafe benchmark defined
- [x] row-level evidence citations implemented
- [x] source/execution/evidence metrics separated
- [x] regression tests added
- [x] full repository CI + SQL evaluator passes
- [x] persisted JSON/Markdown results reviewed
- [x] representative failure cases written down
- [x] ROADMAP marks SQL / Structured RAG DONE only after the final gate passes

Research context: Spider (arXiv:1809.08887) popularized cross-domain text-to-SQL evaluation, while BIRD (arXiv:2305.03111) emphasizes larger databases and realistic schema/value reasoning. This lab is intentionally much smaller and mechanism-focused; it does not claim general text-to-SQL ability.

SQL / Structured RAG satisfies the sub-lab evaluation contract and is eligible to merge; M08 overall remains IN PROGRESS.
