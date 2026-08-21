from __future__ import annotations

import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from rag_practice.production.serving import (
    GuardedServingIndex,
    MutableLexicalIndex,
    QueryResponse,
    ServingDocument,
    UnsafeServingIndex,
    parse_document,
)


def load_production_benchmark(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _response_matches(operation: dict[str, object], response: QueryResponse) -> bool:
    expected_ids = [str(item) for item in operation.get("expected_ids", [])]
    actual_ids = [document.id for document in response.documents]
    if actual_ids != expected_ids:
        return False
    expected_text = operation.get("expected_text_contains")
    if expected_text is not None:
        if not response.documents:
            return False
        if str(expected_text) not in response.documents[0].text:
            return False
    return True


def _is_stale(document: ServingDocument, *, clock: datetime, max_age_days: int) -> bool:
    return document.updated_at < clock - timedelta(days=max_age_days)


def evaluate_serving_system(
    payload: dict[str, object],
    *,
    guarded: bool,
) -> dict[str, object]:
    clock = datetime.fromisoformat(str(payload["clock"]))
    max_age_days = int(payload["max_age_days"])
    initial = [parse_document(item) for item in payload["initial_documents"]]
    if guarded:
        system: GuardedServingIndex | UnsafeServingIndex = GuardedServingIndex(
            initial, clock=clock, max_age_days=max_age_days, require_trusted=True
        )
        name = "guarded"
    else:
        system = UnsafeServingIndex(initial)
        name = "unsafe_baseline"

    query_records: list[dict[str, object]] = []
    mutation_records: list[dict[str, object]] = []
    result_correct = 0
    cache_correct = 0
    cache_labeled = 0
    invalidation_correct = 0
    invalidation_count = 0
    no_evidence_correct = 0
    no_evidence_count = 0
    unauthorized_exposures = 0
    stale_exposures = 0
    untrusted_exposures = 0
    observability_complete = 0
    query_count = 0

    required_trace_fields = {
        "cache_hit",
        "index_generation",
        "candidate_count",
        "acl_filtered",
        "stale_filtered",
        "untrusted_filtered",
        "returned_ids",
        "latency_ms",
    }

    for raw_operation in payload["operations"]:
        operation = dict(raw_operation)
        operation_type = str(operation["type"])
        if operation_type == "upsert":
            document = parse_document(operation["document"])
            latency_ms = system.upsert(document)
            mutation_records.append(
                {
                    "id": operation["id"],
                    "type": "upsert",
                    "document_id": document.id,
                    "generation": system.index.generation,
                    "latency_ms": latency_ms,
                    "success": system.index.documents.get(document.id) == document,
                }
            )
            continue
        if operation_type == "delete":
            deleted, latency_ms = system.delete(str(operation["document_id"]))
            mutation_records.append(
                {
                    "id": operation["id"],
                    "type": "delete",
                    "document_id": operation["document_id"],
                    "generation": system.index.generation,
                    "latency_ms": latency_ms,
                    "success": deleted and str(operation["document_id"]) not in system.index.documents,
                }
            )
            continue
        if operation_type != "query":
            raise ValueError(f"unknown operation type: {operation_type}")

        query_count += 1
        roles = tuple(str(role) for role in operation["roles"])
        response = system.query(str(operation["query"]), roles=roles, k=1)
        result_matches = _response_matches(operation, response)
        result_correct += int(result_matches)

        expected_cache = operation.get("expected_cache_hit_guarded")
        cache_matches: bool | None = None
        if expected_cache is not None:
            cache_labeled += 1
            cache_matches = response.trace.cache_hit == bool(expected_cache)
            cache_correct += int(cache_matches)

        if not operation.get("expected_ids", []):
            no_evidence_count += 1
            no_evidence_correct += int(not response.documents)

        operation_class = str(operation.get("class", ""))
        if operation_class.startswith("cache_invalidation_after_"):
            invalidation_count += 1
            ok = result_matches and not response.trace.cache_hit
            invalidation_correct += int(ok)

        role_set = set(roles)
        unauthorized = any(not (role_set & set(document.roles)) for document in response.documents)
        stale = any(
            _is_stale(document, clock=clock, max_age_days=max_age_days)
            for document in response.documents
        )
        untrusted = any(not document.trusted for document in response.documents)
        unauthorized_exposures += int(unauthorized)
        stale_exposures += int(stale)
        untrusted_exposures += int(untrusted)

        trace_dict = {
            "cache_hit": response.trace.cache_hit,
            "index_generation": response.trace.index_generation,
            "candidate_count": response.trace.candidate_count,
            "acl_filtered": response.trace.acl_filtered,
            "stale_filtered": response.trace.stale_filtered,
            "untrusted_filtered": response.trace.untrusted_filtered,
            "returned_ids": list(response.trace.returned_ids),
            "latency_ms": response.trace.latency_ms,
        }
        observability_complete += int(required_trace_fields <= trace_dict.keys())
        query_records.append(
            {
                "id": operation["id"],
                "class": operation_class,
                "query": operation["query"],
                "roles": list(roles),
                "expected_ids": operation.get("expected_ids", []),
                "actual_ids": [document.id for document in response.documents],
                "actual_texts": [document.text for document in response.documents],
                "result_correct": result_matches,
                "cache_expectation_correct": cache_matches,
                "unauthorized_exposure": unauthorized,
                "stale_exposure": stale,
                "untrusted_exposure": untrusted,
                "trace": trace_dict,
            }
        )

    mean_query_latency = (
        sum(float(record["trace"]["latency_ms"]) for record in query_records) / query_count
        if query_count
        else 0.0
    )
    mean_mutation_latency = (
        sum(float(record["latency_ms"]) for record in mutation_records) / len(mutation_records)
        if mutation_records
        else 0.0
    )
    mutation_accuracy = (
        sum(int(bool(record["success"])) for record in mutation_records) / len(mutation_records)
        if mutation_records
        else 0.0
    )
    return {
        "system": name,
        "scenario_result_accuracy": result_correct / query_count if query_count else 0.0,
        "cache_expectation_accuracy": cache_correct / cache_labeled if cache_labeled else 0.0,
        "cache_invalidation_accuracy": invalidation_correct / invalidation_count
        if invalidation_count
        else 0.0,
        "mutation_accuracy": mutation_accuracy,
        "no_evidence_accuracy": no_evidence_correct / no_evidence_count
        if no_evidence_count
        else 0.0,
        "unauthorized_exposure_rate": unauthorized_exposures / query_count if query_count else 0.0,
        "stale_exposure_rate": stale_exposures / query_count if query_count else 0.0,
        "untrusted_exposure_rate": untrusted_exposures / query_count if query_count else 0.0,
        "observability_completeness": observability_complete / query_count if query_count else 0.0,
        "cache_hit_rate": sum(int(bool(record["trace"]["cache_hit"])) for record in query_records)
        / query_count
        if query_count
        else 0.0,
        "mean_query_latency_ms": mean_query_latency,
        "mean_mutation_latency_ms": mean_mutation_latency,
        "final_index_generation": system.index.generation,
        "final_posting_entries": system.index.posting_entries,
        "queries": query_records,
        "mutations": mutation_records,
    }


def evaluate_scale(payload: dict[str, object]) -> list[dict[str, object]]:
    contract = payload["scale_contract"]
    seed = int(contract["seed"])
    target_id = str(contract["target_id"])
    target_text = str(contract["target_text"])
    query = str(contract["query"])
    records: list[dict[str, object]] = []
    for size in contract["sizes"]:
        count = int(size)
        rng = random.Random(seed + count)
        documents = [
            ServingDocument(
                id=f"scale-{index}",
                text=f"Synthetic corpus record {index} group {rng.randrange(17)} filler token.",
                roles=("public",),
                updated_at=datetime.fromisoformat(str(payload["clock"])),
                trusted=True,
                source_version=1,
            )
            for index in range(count - 1)
        ]
        documents.append(
            ServingDocument(
                id=target_id,
                text=target_text,
                roles=("public",),
                updated_at=datetime.fromisoformat(str(payload["clock"])),
                trusted=True,
                source_version=1,
            )
        )
        started = time.perf_counter()
        index = MutableLexicalIndex(documents)
        build_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        result = index.rank(query, sorted(index.documents), k=1)
        query_ms = (time.perf_counter() - started) * 1000.0

        update = ServingDocument(
            id="scale-update",
            text="Scale update marker indigo meadow.",
            roles=("public",),
            updated_at=datetime.fromisoformat(str(payload["clock"])),
            trusted=True,
            source_version=1,
        )
        started = time.perf_counter()
        index.upsert(update)
        upsert_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        deleted = index.delete(update.id)
        delete_ms = (time.perf_counter() - started) * 1000.0
        records.append(
            {
                "size": count,
                "target_hit@1": result == [target_id],
                "build_ms": build_ms,
                "query_ms": query_ms,
                "upsert_ms": upsert_ms,
                "delete_ms": delete_ms,
                "update_deleted": deleted,
                "posting_entries": index.posting_entries,
            }
        )
    return records
