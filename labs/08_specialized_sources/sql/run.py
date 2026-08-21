from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.sql_structured import (
    evaluate_flat_row_baseline,
    evaluate_structured_traces,
    evidence_complete,
    normalize_answer,
)
from rag_practice.structured_sql import SQLiteStructuredSource, StructuredSQLRAG


ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "m08_sql"
RESULTS = Path(__file__).resolve().parent / "results"


def load_cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (BENCH / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    cases = load_cases()
    case_map = {str(case["id"]): case for case in cases}
    source = SQLiteStructuredSource.from_scripts(
        (BENCH / "schema.sql").read_text(),
        (BENCH / "data.sql").read_text(),
    )
    pipeline = StructuredSQLRAG(source)

    flat_rankings: dict[str, list[str]] = {}
    flat_qrels: dict[str, list[str]] = {}
    traces = {}
    per_query = []

    for case in cases:
        qid = str(case["id"])
        question = str(case["question"])
        if case.get("evidence"):
            flat_hits = source.search(question, limit=5)
            flat_rankings[qid] = [hit.record.id for hit in flat_hits]
            flat_qrels[qid] = list(case["evidence"])
        else:
            flat_rankings[qid] = []

        trace = pipeline.run(question)
        traces[qid] = trace
        relevant = list(case.get("evidence", []))
        evidence_recall = (
            len(set(trace.evidence_ids) & set(relevant)) / len(set(relevant))
            if relevant else None
        )
        answer_correct = (
            normalize_answer(trace.answer) == normalize_answer(str(case["answer"]))
            if not case.get("unsafe") and not case.get("unsupported")
            else None
        )
        per_query.append(
            {
                "id": qid,
                "task": case["task"],
                "question": question,
                "expected_answer": case["answer"],
                "status": trace.status,
                "answer": trace.answer,
                "answer_correct": answer_correct,
                "expected_tables": case.get("tables", []),
                "planned_tables": list(trace.plan.tables) if trace.plan else [],
                "sql": trace.plan.sql.strip() if trace.plan else "",
                "evidence_ids": list(trace.evidence_ids),
                "expected_evidence": relevant,
                "evidence_recall": evidence_recall,
                "evidence_complete": (
                    evidence_complete(trace.evidence_ids, relevant) if relevant else None
                ),
                "citations": list(trace.citations),
                "flat_row_bm25": flat_rankings[qid],
                "error": trace.error,
                "planning_ms": trace.planning_ms,
                "execution_ms": trace.execution_ms,
                "evidence_ms": trace.evidence_ms,
                "end_to_end_ms": trace.end_to_end_ms,
            }
        )

    flat_metrics = evaluate_flat_row_baseline(flat_rankings, flat_qrels, k=5)
    sql_metrics = evaluate_structured_traces(traces, case_map)

    result = {
        "hypothesis": (
            "structured RAG should preserve schema/execution semantics and row provenance; "
            "flat lexical row retrieval cannot reliably answer joins/aggregates, while SQL "
            "must fail closed on unsafe or unsupported requests"
        ),
        "benchmark": {
            "tables": len(source.schema()),
            "rows": sum(table.row_count for table in source.schema().values()),
            "queries": len(cases),
            "safe_queries": sum(not c.get("unsafe") and not c.get("unsupported") for c in cases),
            "unsafe_queries": sum(bool(c.get("unsafe")) for c in cases),
            "unsupported_queries": sum(bool(c.get("unsupported")) for c in cases),
        },
        "systems": {
            "flat_row_bm25": {"metrics": flat_metrics},
            "schema_aware_validated_sql": {"metrics": sql_metrics},
        },
        "per_query": per_query,
        "guardrails": [
            "qrels/reference answers are used only after planning/execution",
            "SQL is parameterized where user entities become values",
            "validator allows only SELECT/WITH and SQLite runs with PRAGMA query_only=ON",
            "aggregate answers persist row-level evidence locators separately from result rows",
            "rule-based planning is a controlled teaching mechanism, not a general text-to-SQL claim",
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# M08.2 SQL / Structured RAG results",
        "",
        f"Benchmark: {result['benchmark']['tables']} tables, {result['benchmark']['rows']} rows, {result['benchmark']['queries']} queries.",
        "",
        "| System | Evidence recall | Evidence complete | Answer exact | Execution success | Unsafe reject | Empty correct | Unsupported handled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| flat row BM25@5 | {flat_metrics['evidence_recall@5']:.3f} | "
            f"{flat_metrics['evidence_complete@5']:.3f} | n/a | n/a | n/a | n/a | n/a |"
        ),
        (
            f"| schema-aware validated SQL | {sql_metrics['evidence_recall']:.3f} | "
            f"{sql_metrics['evidence_complete_rate']:.3f} | "
            f"{sql_metrics['answer_exact_match']:.3f} | "
            f"{sql_metrics['execution_success_rate']:.3f} | "
            f"{sql_metrics['unsafe_rejection_rate']:.3f} | "
            f"{sql_metrics['empty_result_accuracy']:.3f} | "
            f"{sql_metrics['unsupported_handling_rate']:.3f} |"
        ),
        "",
        "## Interpretation guardrails",
        "",
        "- Flat BM25 is a retrieval-only control; it cannot compute joins or aggregates.",
        "- SQL answer correctness is evaluated separately from row-level evidence completeness.",
        "- Unsafe mutation and unsupported-schema requests are expected to fail closed.",
        "- SQLite latency is only a deterministic sanity measurement, not a production database claim.",
    ]
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
