import json
from datetime import datetime
from pathlib import Path

from rag_practice.evaluation.production import evaluate_serving_system
from rag_practice.production.serving import (
    GuardedServingIndex,
    MutableLexicalIndex,
    ServingDocument,
    UnsafeServingIndex,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "m10_production" / "scenarios.json"


def _document(identifier: str, text: str, *, roles=("public",), version=1) -> ServingDocument:
    return ServingDocument(
        id=identifier,
        text=text,
        roles=roles,
        updated_at=datetime.fromisoformat("2026-07-31T00:00:00+00:00"),
        trusted=True,
        source_version=version,
    )


def test_mutable_index_upsert_delete_updates_postings_without_full_rebuild_contract() -> None:
    index = MutableLexicalIndex([_document("d1", "alpha blue pine")])
    assert index.rank("blue pine", ["d1"], k=1) == ["d1"]
    first_generation = index.generation
    index.upsert(_document("d1", "alpha green cedar", version=2))
    assert index.generation == first_generation + 1
    assert index.rank("blue pine", ["d1"], k=1) == []
    assert index.rank("green cedar", ["d1"], k=1) == ["d1"]
    assert index.delete("d1") is True
    assert index.rank("green cedar", ["d1"], k=1) == []


def test_guarded_cache_is_role_and_generation_aware() -> None:
    clock = datetime.fromisoformat("2026-08-01T00:00:00+00:00")
    public = _document("public", "Atlas code blue pine")
    private = _document("private", "Finance code gold ledger", roles=("finance",))
    system = GuardedServingIndex([public, private], clock=clock, max_age_days=30)

    first = system.query("Atlas code", roles=("public",))
    second = system.query("Atlas code", roles=("public",))
    assert first.trace.cache_hit is False
    assert second.trace.cache_hit is True

    denied = system.query("Finance code", roles=("public",))
    allowed = system.query("Finance code", roles=("finance",))
    assert denied.documents == ()
    assert [document.id for document in allowed.documents] == ["private"]

    system.upsert(_document("public", "Atlas code green cedar", version=2))
    refreshed = system.query("Atlas code", roles=("public",))
    assert refreshed.trace.cache_hit is False
    assert "green cedar" in refreshed.documents[0].text


def test_unsafe_cache_retains_old_snapshot_after_mutation() -> None:
    system = UnsafeServingIndex([_document("d1", "Atlas code blue pine")])
    first = system.query("Atlas code", roles=("public",))
    assert first.trace.cache_hit is False
    system.upsert(_document("d1", "Atlas code green cedar", version=2))
    stale = system.query("Atlas code", roles=("public",))
    assert stale.trace.cache_hit is True
    assert "blue pine" in stale.documents[0].text


def test_frozen_production_workload_separates_unsafe_and_guarded_outcomes() -> None:
    payload = json.loads(BENCHMARK.read_text())
    guarded = evaluate_serving_system(payload, guarded=True)
    unsafe = evaluate_serving_system(payload, guarded=False)
    assert guarded["scenario_result_accuracy"] == 1.0
    assert guarded["cache_invalidation_accuracy"] == 1.0
    assert guarded["unauthorized_exposure_rate"] == 0.0
    assert guarded["stale_exposure_rate"] == 0.0
    assert guarded["untrusted_exposure_rate"] == 0.0
    assert guarded["observability_completeness"] == 1.0
    assert unsafe["scenario_result_accuracy"] < guarded["scenario_result_accuracy"]
    assert unsafe["cache_invalidation_accuracy"] < 1.0
    assert unsafe["unauthorized_exposure_rate"] > 0.0
    assert unsafe["stale_exposure_rate"] > 0.0
    assert unsafe["untrusted_exposure_rate"] > 0.0
