import json
from pathlib import Path

from rag_practice.metadata_filter import FilterAwareBM25, FilterPredicate, FilterRequest
from rag_practice.sources.base import SourceRecord

ROOT = Path(__file__).resolve().parents[1]


def load_records() -> dict[str, SourceRecord]:
    records = {}
    for line in (ROOT / "benchmarks/m08_metadata/records.jsonl").read_text().splitlines():
        row = json.loads(line)
        records[row["id"]] = SourceRecord(
            id=row["id"],
            source_type="metadata_document",
            locator=f"memory://metadata/{row['id']}",
            title=row["title"],
            content=row["content"],
            metadata=row["metadata"],
        )
    return records


def test_security_predicate_enforces_tenant_and_role() -> None:
    records = load_records()
    viewer = FilterRequest(tenant="alpha", role="viewer")
    assert FilterPredicate.security_allowed(records["a2"], viewer)
    assert not FilterPredicate.security_allowed(records["a3"], viewer)
    assert not FilterPredicate.security_allowed(records["b3"], viewer)
    assert FilterPredicate.security_allowed(records["x1"], viewer)


def test_explicit_filters_use_and_semantics() -> None:
    records = load_records()
    request = FilterRequest(
        tenant="alpha", role="analyst", product="nimbusdb", region="eu"
    )
    assert FilterPredicate.matches(records["a4"], request)
    assert not FilterPredicate.matches(records["a9"], request)
    assert not FilterPredicate.matches(records["b4"], request)


def test_unfiltered_baseline_can_leak_wrong_tenant() -> None:
    retriever = FilterAwareBM25(load_records())
    trace = retriever.search(
        "NimbusDB encryption key rotation procedure",
        FilterRequest(tenant="alpha", role="admin"),
        strategy="unfiltered",
        limit=3,
    )
    assert trace.hits[0].record.id == "b1"
    assert not FilterPredicate.security_allowed(trace.hits[0].record, FilterRequest(tenant="alpha", role="admin"))


def test_small_postfilter_budget_can_lose_authorized_relevant_doc() -> None:
    retriever = FilterAwareBM25(load_records())
    request = FilterRequest(tenant="alpha", role="admin")
    post = retriever.search(
        "NimbusDB encryption key rotation procedure",
        request,
        strategy="postfilter",
        limit=3,
        candidate_limit=2,
    )
    pre = retriever.search(
        "NimbusDB encryption key rotation procedure",
        request,
        strategy="prefilter",
        limit=3,
    )
    assert post.hits == ()
    assert pre.hits[0].record.id == "a1"


def test_prefilter_never_returns_constraint_violations() -> None:
    retriever = FilterAwareBM25(load_records())
    request = FilterRequest(
        tenant="alpha", role="developer", product="aurora", region="apac"
    )
    trace = retriever.search(
        "Aurora deployment endpoint", request, strategy="prefilter", limit=3
    )
    assert trace.hits
    assert all(FilterPredicate.matches(hit.record, request) for hit in trace.hits)
    assert trace.hits[0].record.id == "a7"


def test_empty_filter_returns_no_results() -> None:
    trace = FilterAwareBM25(load_records()).search(
        "NimbusDB compliance controls",
        FilterRequest(
            tenant="alpha", role="viewer", product="nimbusdb", region="latam"
        ),
        strategy="prefilter",
        limit=3,
    )
    assert trace.hits == ()
    assert trace.eligible_records == 0
