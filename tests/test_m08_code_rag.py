from __future__ import annotations

import json
from pathlib import Path

from rag_practice.code_rag import PythonRepositoryIndex
from rag_practice.evaluation.code_rag import evaluate_code_rankings


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m08_code"


def _index() -> PythonRepositoryIndex:
    return PythonRepositoryIndex(BENCHMARK / "repo")


def _queries() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (BENCHMARK / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_ast_index_preserves_symbols_and_exact_line_locators() -> None:
    index = _index()

    assert len(index.files) == 13
    assert len(index.symbols) == 17

    invalidate = index.symbols["storage/cache.py::Cache.invalidate"]
    assert invalidate.kind == "method"
    assert invalidate.line == 10
    assert invalidate.end_line == 12
    assert invalidate.locator == "code://repo/storage/cache.py#L10-L12"


def test_duplicate_symbol_names_remain_distinct_repository_locations() -> None:
    index = _index()

    assert set(index.by_name["parse_token"]) == {
        "auth/tokens.py::parse_token",
        "billing/tokens.py::parse_token",
    }
    assert set(index.by_name["normalize"]) == {
        "billing/normalizers.py::normalize",
        "helpers/normalizers.py::normalize",
    }


def test_call_graph_resolves_imported_functions_but_not_local_attributes() -> None:
    index = _index()

    assert index.call_graph["pricing/engine.py::compute_total"] == [
        "pricing/discounts.py::apply_discount",
        "pricing/tax.py::tax_for_region",
    ]
    assert index.call_graph["services/checkout.py::finalize_checkout"] == [
        "pricing/engine.py::compute_total",
        "payments/gateway.py::charge",
    ]
    assert index.reverse_call_graph["domain/orders.py::create_order"] == [
        "api/orders.py::submit_order"
    ]

    # `rates.get(...)` in tax_for_region is a local dict method call. It must not be
    # mis-resolved to the unrelated Cache.get method merely because `get` is unique.
    assert index.call_graph["pricing/tax.py::tax_for_region"] == []
    assert "storage/cache.py::Cache.get" not in {
        target for targets in index.call_graph.values() for target in targets
    }


def test_symbol_aware_routing_prefers_implementation_over_call_site() -> None:
    index = _index()

    compute = index.search_symbol_graph(
        "Find the implementation of compute_total, not the checkout call site.", k=4
    )
    create = index.search_symbol_graph(
        "Where is create_order implemented rather than the API submit_order wrapper?",
        k=4,
    )

    assert compute[0][0] == "pricing/engine.py::compute_total"
    assert create[0][0] == "domain/orders.py::create_order"


def test_graph_expansion_recovers_dependencies_and_reverse_change_locality() -> None:
    index = _index()

    checkout = {
        symbol_id
        for symbol_id, _ in index.search_symbol_graph(
            "Which functions does finalize_checkout call before returning?", k=4
        )
    }
    rename = index.search_symbol_graph(
        "If apply_discount is renamed, which caller must change?", k=4
    )

    assert {
        "pricing/engine.py::compute_total",
        "payments/gateway.py::charge",
    }.issubset(checkout)
    assert rename[0][0] == "pricing/engine.py::compute_total"
    assert "pricing/discounts.py::apply_discount" in {item[0] for item in rename}


def test_symbol_graph_exposes_shared_source_contract() -> None:
    index = _index()
    hits = index.search("Where is bearer auth token validation implemented?", limit=2)

    assert hits[0].record.id == "auth/tokens.py::validate_token"
    assert hits[0].record.source_type == "code"
    assert hits[0].record.locator.startswith("code://repo/auth/tokens.py#L")
    assert hits[0].details["retrieval"] == "symbol_graph"


def test_evaluation_keeps_file_hits_separate_from_exact_symbol_locations() -> None:
    index = _index()
    queries = _queries()

    file_rankings = {
        str(query["id"]): [item[0] for item in index.search_files(str(query["query"]), k=4)]
        for query in queries
    }
    symbol_rankings = {
        str(query["id"]): [
            item[0] for item in index.search_symbols(str(query["query"]), k=4)
        ]
        for query in queries
    }
    graph_rankings = {
        str(query["id"]): [
            item[0] for item in index.search_symbol_graph(str(query["query"]), k=4)
        ]
        for query in queries
    }
    zero_latency = {str(query["id"]): 0.0 for query in queries}

    file_result = evaluate_code_rankings(
        index,
        queries,
        system="file_bm25",
        rankings=file_rankings,
        latencies_ms=zero_latency,
    )
    symbol_result = evaluate_code_rankings(
        index,
        queries,
        system="symbol_bm25",
        rankings=symbol_rankings,
        latencies_ms=zero_latency,
    )
    graph_result = evaluate_code_rankings(
        index,
        queries,
        system="symbol_graph",
        rankings=graph_rankings,
        latencies_ms=zero_latency,
    )

    assert file_result["metrics"]["exact_line_locators_available"] == 0.0
    assert symbol_result["metrics"]["exact_line_locators_available"] == 1.0
    assert graph_result["metrics"]["single_evidence_answer_location_exact"] == 1.0
    assert graph_result["metrics"]["dependency_complete@4"] == 1.0

    # Preserve the useful negative result: changing from files to isolated symbols
    # can lose dependency evidence before graph expansion recovers it.
    assert (
        symbol_result["metrics"]["evidence_complete@4"]
        < file_result["metrics"]["evidence_complete@4"]
    )
    assert (
        graph_result["metrics"]["evidence_complete@4"]
        > symbol_result["metrics"]["evidence_complete@4"]
    )
