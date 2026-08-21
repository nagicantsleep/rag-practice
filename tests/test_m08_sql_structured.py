from pathlib import Path

from rag_practice.structured_sql import (
    RuleBasedSQLPlanner,
    SQLReadOnlyValidator,
    SQLiteStructuredSource,
    StructuredSQLRAG,
)

ROOT = Path(__file__).resolve().parents[1]


def make_source() -> SQLiteStructuredSource:
    return SQLiteStructuredSource.from_scripts(
        (ROOT / "benchmarks/m08_sql/schema.sql").read_text(),
        (ROOT / "benchmarks/m08_sql/data.sql").read_text(),
    )


def test_schema_discovery_and_row_locators() -> None:
    source = make_source()
    schema = source.schema()
    assert set(schema) == {"customers", "order_items", "orders", "products"}
    assert schema["order_items"].primary_key == ("order_id", "product_id")
    assert source.get_record("order_items:1001:101").locator == (
        "sqlite://commerce/order_items/1001/101"
    )


def test_shared_source_contract_flat_row_search() -> None:
    hits = make_source().search("Cora Labs North region", limit=3)
    assert hits
    assert hits[0].record.source_type == "sqlite_row"


def test_validator_rejects_mutation_and_multi_statement() -> None:
    source = make_source()
    schema = source.schema()
    validator = SQLReadOnlyValidator()
    planner = RuleBasedSQLPlanner()
    delete_plan = planner.plan(
        "Delete cancelled orders and tell me how many remain.", schema
    )
    valid, _ = validator.validate(delete_plan, schema)
    assert not valid

    from rag_practice.structured_sql.planner import StructuredQueryPlan

    multi = StructuredQueryPlan(
        "bad", "SELECT 1; SELECT 2", (), "", (), (), operation="select"
    )
    valid, _ = validator.validate(multi, schema)
    assert not valid


def test_pipeline_executes_aggregate_with_row_level_provenance() -> None:
    trace = StructuredSQLRAG(make_source()).run(
        "What was the total shipped revenue in 2026?"
    )
    assert trace.status == "ok"
    assert trace.answer == "730"
    assert set(trace.evidence_ids) == {
        "order_items:1001:101",
        "order_items:1001:103",
        "order_items:1002:102",
        "order_items:1004:101",
        "order_items:1004:102",
    }
    assert all(citation.startswith("sqlite://commerce/order_items/") for citation in trace.citations)


def test_pipeline_handles_empty_result_without_fabricating_evidence() -> None:
    trace = StructuredSQLRAG(make_source()).run(
        "Which shipped orders went to the Antarctica region in 2026?"
    )
    assert trace.status == "ok"
    assert trace.answer == "NO_ROWS"
    assert trace.evidence_ids == ()
    assert trace.citations == ()


def test_unsafe_query_is_rejected_and_database_is_unchanged() -> None:
    source = make_source()
    before = source.execute_readonly("SELECT COUNT(*) FROM orders").rows
    trace = StructuredSQLRAG(source).run(
        "Delete cancelled orders and tell me how many remain."
    )
    after = source.execute_readonly("SELECT COUNT(*) FROM orders").rows
    assert trace.status == "rejected"
    assert before == after == ((5,),)


def test_unsupported_question_fails_closed() -> None:
    trace = StructuredSQLRAG(make_source()).run(
        "What loyalty tier is attached to every order?"
    )
    assert trace.status == "planning_error"
    assert trace.rows == ()
    assert trace.citations == ()
