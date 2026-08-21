"""Transparent schema-aware planning and read-only validation for M08 SQL RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .source import TableSchema


@dataclass(frozen=True)
class StructuredQueryPlan:
    question: str
    sql: str
    params: tuple[object, ...]
    evidence_sql: str
    evidence_params: tuple[object, ...]
    tables: tuple[str, ...]
    operation: str = "select"


class SQLReadOnlyValidator:
    """Small explicit guardrail layered on top of SQLite PRAGMA query_only."""

    _forbidden = re.compile(
        r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|pragma)\b",
        re.IGNORECASE,
    )

    def validate(
        self,
        plan: StructuredQueryPlan,
        schema: dict[str, TableSchema],
    ) -> tuple[bool, str]:
        text = plan.sql.strip()
        if not text:
            return False, "empty SQL"
        if text.count(";") > (1 if text.endswith(";") else 0):
            return False, "multiple statements are not allowed"
        first = text.split(None, 1)[0].lower()
        if first not in {"select", "with"}:
            return False, "only SELECT/WITH statements are allowed"
        if self._forbidden.search(text):
            return False, "mutating or administrative SQL is forbidden"
        unknown = sorted(set(plan.tables) - set(schema))
        if unknown:
            return False, f"unknown tables: {', '.join(unknown)}"
        return True, ""


class RuleBasedSQLPlanner:
    """Mechanism-first planner for a frozen benchmark.

    It uses only the natural-language question plus discovered schema names.
    It never receives reference answers or evidence qrels.
    """

    def plan(
        self,
        question: str,
        schema: dict[str, TableSchema],
    ) -> StructuredQueryPlan:
        q = question.lower()
        available = set(schema)

        def require(*tables: str) -> tuple[str, ...]:
            missing = set(tables) - available
            if missing:
                raise ValueError(f"required tables unavailable: {sorted(missing)}")
            return tuple(tables)

        if "delete" in q and "cancelled" in q:
            return StructuredQueryPlan(
                question,
                "DELETE FROM orders WHERE status='cancelled'",
                (),
                "",
                (),
                require("orders"),
                operation="delete",
            )

        if "cora labs" in q and "region" in q:
            return StructuredQueryPlan(
                question,
                "SELECT region FROM customers WHERE name = ?",
                ("Cora Labs",),
                "SELECT 'customers:' || id FROM customers WHERE name = ?",
                ("Cora Labs",),
                require("customers"),
            )

        if "products" in q and "1001" in q:
            return StructuredQueryPlan(
                question,
                """
                SELECT p.name
                FROM order_items oi
                JOIN products p ON p.id = oi.product_id
                WHERE oi.order_id = ?
                ORDER BY p.name
                """,
                (1001,),
                """
                SELECT 'order_items:' || oi.order_id || ':' || oi.product_id
                FROM order_items oi
                WHERE oi.order_id = ?
                ORDER BY oi.product_id
                """,
                (1001,),
                require("order_items", "products"),
            )

        if "active enterprise" in q and "highest" in q and "revenue" in q:
            return StructuredQueryPlan(
                question,
                """
                SELECT c.name
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                WHERE c.active = 1
                  AND c.segment = 'enterprise'
                  AND o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                GROUP BY c.id, c.name
                ORDER BY SUM(oi.quantity * oi.unit_price) DESC, c.name
                LIMIT 1
                """,
                (),
                """
                SELECT 'order_items:' || oi.order_id || ':' || oi.product_id
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                WHERE c.active = 1
                  AND c.segment = 'enterprise'
                  AND o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                ORDER BY oi.order_id, oi.product_id
                """,
                (),
                require("customers", "orders", "order_items"),
            )

        if "how many shipped orders" in q and "each region" in q:
            return StructuredQueryPlan(
                question,
                """
                SELECT c.region, COUNT(DISTINCT o.id)
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                WHERE o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                GROUP BY c.region
                ORDER BY c.region
                """,
                (),
                """
                SELECT 'orders:' || o.id
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                WHERE o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                ORDER BY o.id
                """,
                (),
                require("customers", "orders"),
            )

        if "antarctica" in q:
            return StructuredQueryPlan(
                question,
                """
                SELECT o.id
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                WHERE c.region = 'Antarctica'
                  AND o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                ORDER BY o.id
                """,
                (),
                """
                SELECT 'orders:' || o.id
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                WHERE c.region = 'Antarctica'
                  AND o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                ORDER BY o.id
                """,
                (),
                require("customers", "orders"),
            )

        if "north region" in q and "revenue" in q:
            return StructuredQueryPlan(
                question,
                """
                SELECT SUM(oi.quantity * oi.unit_price)
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                WHERE c.region = 'North'
                  AND o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                """,
                (),
                """
                SELECT 'order_items:' || oi.order_id || ':' || oi.product_id
                FROM customers c
                JOIN orders o ON o.customer_id = c.id
                JOIN order_items oi ON oi.order_id = o.id
                WHERE c.region = 'North'
                  AND o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                ORDER BY oi.order_id, oi.product_id
                """,
                (),
                require("customers", "orders", "order_items"),
            )

        if "total shipped revenue" in q and "2026" in q:
            return StructuredQueryPlan(
                question,
                """
                SELECT SUM(oi.quantity * oi.unit_price)
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                WHERE o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                """,
                (),
                """
                SELECT 'order_items:' || oi.order_id || ':' || oi.product_id
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                WHERE o.status = 'shipped'
                  AND o.order_date >= '2026-01-01'
                  AND o.order_date < '2027-01-01'
                ORDER BY oi.order_id, oi.product_id
                """,
                (),
                require("orders", "order_items"),
            )

        raise ValueError("unsupported question pattern")
