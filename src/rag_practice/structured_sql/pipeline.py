"""Execution pipeline that keeps planning, validation, SQL, evidence, and answer traces separate."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .planner import RuleBasedSQLPlanner, SQLReadOnlyValidator, StructuredQueryPlan
from .source import SQLiteStructuredSource


@dataclass(frozen=True)
class StructuredSQLTrace:
    question: str
    status: str
    plan: StructuredQueryPlan | None
    answer: str
    rows: tuple[tuple[object, ...], ...]
    evidence_ids: tuple[str, ...]
    citations: tuple[str, ...]
    schema_tables: tuple[str, ...]
    planning_ms: float
    execution_ms: float
    evidence_ms: float
    end_to_end_ms: float
    error: str = ""


class StructuredSQLRAG:
    def __init__(
        self,
        source: SQLiteStructuredSource,
        *,
        planner: RuleBasedSQLPlanner | None = None,
        validator: SQLReadOnlyValidator | None = None,
    ) -> None:
        self.source = source
        self.planner = planner or RuleBasedSQLPlanner()
        self.validator = validator or SQLReadOnlyValidator()

    @staticmethod
    def _format_answer(rows: tuple[tuple[object, ...], ...]) -> str:
        if not rows:
            return "NO_ROWS"
        if len(rows) == 1 and len(rows[0]) == 1:
            value = rows[0][0]
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)
        if all(len(row) == 1 for row in rows):
            return "; ".join(str(row[0]) for row in rows)
        if all(len(row) == 2 for row in rows):
            return "; ".join(f"{row[0]}={row[1]}" for row in rows)
        return "; ".join(" | ".join(str(value) for value in row) for row in rows)

    def run(self, question: str) -> StructuredSQLTrace:
        started = perf_counter()
        schema = self.source.schema()

        planning_started = perf_counter()
        try:
            plan = self.planner.plan(question, schema)
        except ValueError as exc:
            planning_ms = (perf_counter() - planning_started) * 1000.0
            return StructuredSQLTrace(
                question, "planning_error", None, "", (), (), (), tuple(schema),
                planning_ms, 0.0, 0.0, (perf_counter() - started) * 1000.0, str(exc)
            )
        planning_ms = (perf_counter() - planning_started) * 1000.0

        valid, error = self.validator.validate(plan, schema)
        if not valid:
            return StructuredSQLTrace(
                question, "rejected", plan, "", (), (), (), tuple(schema),
                planning_ms, 0.0, 0.0, (perf_counter() - started) * 1000.0, error
            )

        execution = self.source.execute_readonly(plan.sql, plan.params)
        evidence_ids: tuple[str, ...] = ()
        evidence_ms = 0.0
        citations: tuple[str, ...] = ()
        if plan.evidence_sql:
            evidence = self.source.execute_readonly(plan.evidence_sql, plan.evidence_params)
            evidence_ids = tuple(str(row[0]) for row in evidence.rows)
            evidence_ms = evidence.latency_ms
            citations = tuple(
                self.source.get_record(record_id).locator for record_id in evidence_ids
            )

        return StructuredSQLTrace(
            question=question,
            status="ok",
            plan=plan,
            answer=self._format_answer(execution.rows),
            rows=execution.rows,
            evidence_ids=evidence_ids,
            citations=citations,
            schema_tables=tuple(schema),
            planning_ms=planning_ms,
            execution_ms=execution.latency_ms,
            evidence_ms=evidence_ms,
            end_to_end_ms=(perf_counter() - started) * 1000.0,
        )
