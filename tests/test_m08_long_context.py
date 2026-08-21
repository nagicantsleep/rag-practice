from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rag_practice.evaluation.long_context import mechanism_suite
from rag_practice.long_context import (
    DeterministicEvidenceReader,
    ExplicitLongContextRouter,
    load_benchmark,
    select_context,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m08_long_context" / "benchmark.json"


def test_frozen_long_context_benchmark_integrity() -> None:
    benchmark = load_benchmark(BENCHMARK)

    assert len(benchmark.bundles) == 4
    assert len(benchmark.queries) == 12
    assert benchmark.contract.retrieval_top_k == 2
    assert benchmark.contract.direct_word_threshold == 100
    assert benchmark.contract.global_route_markers == ("across the entire", "list every")
    assert {bundle_id: bundle.word_count for bundle_id, bundle in benchmark.bundles.items()} == {
        "cedar_brief": 46,
        "atlas_handbook": 779,
        "orion_report": 650,
        "lumen_notes": 341,
    }


def test_explicit_router_matches_frozen_preferred_routes_without_qrels() -> None:
    benchmark = load_benchmark(BENCHMARK)
    router = ExplicitLongContextRouter(benchmark.contract)

    for query in benchmark.queries:
        bundle = benchmark.bundles[query.bundle_id]
        assert router.route(query.question, bundle.word_count) == query.preferred_route

        poisoned = replace(
            query,
            relevant=("not-a-real-section",),
            expected_answer="definitely-not-the-answer",
        )
        assert router.route(poisoned.question, bundle.word_count) == query.preferred_route


def test_retrieval_has_preserved_global_evidence_failures() -> None:
    benchmark = load_benchmark(BENCHMARK)
    by_id = {query.id: query for query in benchmark.queries}

    expected = {
        "r5": ({"atlas_s2", "atlas_s11"}, 0.25),
        "r6": ({"atlas_s12", "atlas_s6"}, 2 / 3),
        "r10": ({"orion_s5", "orion_s7"}, 2 / 3),
    }
    for query_id, (section_ids, recall) in expected.items():
        query = by_id[query_id]
        selection = select_context(benchmark, query, route="retrieve")
        assert set(selection.section_ids) == section_ids
        actual = len(set(query.relevant) & set(selection.section_ids)) / len(query.relevant)
        assert actual == recall


def test_deterministic_reader_abstains_when_requested_fact_is_absent() -> None:
    benchmark = load_benchmark(BENCHMARK)
    reader = DeterministicEvidenceReader(abstain_token=benchmark.contract.abstain_token)
    by_id = {query.id: query for query in benchmark.queries}

    for query_id in ("r3", "r7"):
        query = by_id[query_id]
        direct = select_context(benchmark, query, route="direct")
        assert reader.answer(query.question, direct.texts) == "ABSTAIN"


def test_mechanism_suite_records_expected_quality_cost_boundary() -> None:
    benchmark = load_benchmark(BENCHMARK)
    result = mechanism_suite(benchmark)
    systems = {system["system"]: system for system in result["systems"]}

    direct = systems["always_direct"]["metrics"]
    retrieve = systems["always_retrieve"]["metrics"]
    routed = systems["explicit_router"]["metrics"]

    assert direct["answer_accuracy"] == 1.0
    assert direct["evidence_complete"] == 1.0
    assert direct["unnecessary_full_context_rate"] == 1.0

    assert retrieve["answer_accuracy"] == 0.75
    assert retrieve["evidence_recall"] == 0.8583333333333333
    assert retrieve["evidence_complete"] == 0.7
    assert retrieve["mean_context_words"] < direct["mean_context_words"]

    assert routed["route_accuracy"] == 1.0
    assert routed["answer_accuracy"] == 1.0
    assert routed["evidence_complete"] == 1.0
    assert routed["abstention_accuracy"] == 1.0
    assert routed["unnecessary_retrieval_rate"] == 0.0
    assert routed["unnecessary_full_context_rate"] == 0.0
    assert routed["mean_context_words"] < direct["mean_context_words"]
    assert routed["mean_retrieval_calls"] == 5 / 12
