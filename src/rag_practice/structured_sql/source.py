"""SQLite-backed structured source with row-level provenance."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from rag_practice.ir.bm25 import BM25Index
from rag_practice.sources.base import SourceHit, SourceRecord


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class SQLExecution:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    latency_ms: float


class SQLiteStructuredSource:
    """In-memory SQLite source used for deterministic structured-source labs."""

    name = "sqlite-commerce"

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def from_scripts(cls, schema_sql: str, data_sql: str) -> "SQLiteStructuredSource":
        connection = sqlite3.connect(":memory:")
        connection.executescript(schema_sql)
        connection.executescript(data_sql)
        connection.execute("PRAGMA query_only = ON")
        return cls(connection)

    def schema(self) -> dict[str, TableSchema]:
        tables = [
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        result: dict[str, TableSchema] = {}
        for table in tables:
            info = list(self.connection.execute(f'PRAGMA table_info("{table}")'))
            columns = tuple(row[1] for row in info)
            pk = tuple(
                row[1]
                for row in sorted((row for row in info if row[5]), key=lambda row: row[5])
            )
            row_count = int(self.connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            result[table] = TableSchema(table, columns, pk, row_count)
        return result

    def execute_readonly(self, sql: str, params: tuple[object, ...] = ()) -> SQLExecution:
        start = perf_counter()
        cursor = self.connection.execute(sql, params)
        rows = tuple(tuple(row) for row in cursor.fetchall())
        columns = tuple(item[0] for item in cursor.description or ())
        return SQLExecution(columns, rows, (perf_counter() - start) * 1000.0)

    def _row_id(self, table: str, row: sqlite3.Row, schema: TableSchema) -> str:
        key_columns = schema.primary_key or schema.columns[:1]
        key = ":".join(str(row[column]) for column in key_columns)
        return f"{table}:{key}"

    def _row_record(self, table: str, row: sqlite3.Row, schema: TableSchema) -> SourceRecord:
        record_id = self._row_id(table, row, schema)
        content = " ".join(f"{column}={row[column]}" for column in schema.columns)
        return SourceRecord(
            id=record_id,
            source_type="sqlite_row",
            locator="sqlite://commerce/" + record_id.replace(":", "/"),
            title=table,
            content=content,
            metadata={"table": table, "columns": schema.columns},
        )

    def all_row_records(self) -> dict[str, SourceRecord]:
        records: dict[str, SourceRecord] = {}
        for table, schema in self.schema().items():
            rows = self.connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            for row in rows:
                record = self._row_record(table, row, schema)
                records[record.id] = record
        return records

    def get_record(self, record_id: str) -> SourceRecord:
        records = self.all_row_records()
        if record_id not in records:
            raise KeyError(record_id)
        return records[record_id]

    def search(self, query: str, *, limit: int = 5) -> list[SourceHit]:
        """Flat row-BM25 control implementing the shared M08 Source contract."""

        if limit <= 0:
            return []
        records = self.all_row_records()
        corpus = {
            record_id: f"{record.title} {record.content}"
            for record_id, record in records.items()
        }
        index = BM25Index(corpus)
        ranking = index.search(query, k=limit)
        return [
            SourceHit(
                record=records[record_id],
                score=score,
                rank=rank,
                details={"baseline": "flat_row_bm25"},
            )
            for rank, (record_id, score) in enumerate(ranking, start=1)
        ]
