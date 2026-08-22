"""M11.3 generation-aware serving around the frozen integrated copilot."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .integrated import IntegratedCopilot, IntegratedResult


POLICY_VERSION = "m11.2-v1"


@dataclass(frozen=True)
class ServingCacheKey:
    question: str
    user_id: str
    roles: tuple[str, ...]
    snapshot_id: str
    generation: int
    policy_version: str = POLICY_VERSION


@dataclass
class ServingTrace:
    request_sequence: int
    user_id: str
    roles: list[str]
    snapshot_id: str
    generation: int
    cache_hit: bool
    cache_key: dict[str, Any]
    actions: list[dict[str, Any]]
    evidence_ids: list[str]
    source_families: list[str]
    rejected_unauthorized_ids: list[str]
    rejected_stale_ids: list[str]
    rejected_untrusted_ids: list[str]
    stop_reason: str
    action_count: int
    integrated_latency_ms: float
    serving_latency_ms: float
    synthetic_tool_cost: float


@dataclass
class ServingResponse:
    result: IntegratedResult
    trace: ServingTrace


@dataclass
class MutableServingState:
    """Small inspectable state machine for the frozen g0→g1 mutation."""

    active_snapshot: str = "g0"
    generation: int = 0
    appended_event_ids: list[str] = field(default_factory=list)
    replaced_shipment_ids: list[str] = field(default_factory=list)

    def apply_g1_mutation(self) -> None:
        if self.active_snapshot == "g1":
            return
        self.appended_event_ids.append("EV-H003")
        self.replaced_shipment_ids.append("SH-1008@g1")
        self.active_snapshot = "g1"
        self.generation += 1


class GuardedOtcServing:
    """Role/snapshot/generation-aware cache with auditable request traces."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.copilot = IntegratedCopilot(self.data_root)
        self.state = MutableServingState()
        self.cache: dict[ServingCacheKey, IntegratedResult] = {}
        self.request_sequence = 0
        self.scale_records: dict[str, dict[str, Any]] = {}

    def _key(self, question: str, user_id: str, snapshot_id: str) -> ServingCacheKey:
        return ServingCacheKey(
            question=question,
            user_id=user_id,
            roles=tuple(sorted(self.copilot.data.roles(user_id))),
            snapshot_id=snapshot_id,
            generation=self.state.generation,
        )

    @staticmethod
    def _cost(result: IntegratedResult, cache_hit: bool) -> float:
        return 0.05 if cache_hit else float(len(result.actions))

    def query(
        self,
        question: str,
        user_id: str,
        *,
        snapshot_id: str | None = None,
    ) -> ServingResponse:
        started = time.perf_counter()
        snapshot = snapshot_id or self.state.active_snapshot
        key = self._key(question, user_id, snapshot)
        self.request_sequence += 1
        cache_hit = key in self.cache

        if cache_hit:
            result = copy.deepcopy(self.cache[key])
        else:
            result = self.copilot.run(question, user_id, snapshot)
            # Authorization-denied results are deliberately not cached.
            if result.stop_reason != "authorization_denied":
                self.cache[key] = copy.deepcopy(result)

        elapsed = (time.perf_counter() - started) * 1000
        trace = ServingTrace(
            request_sequence=self.request_sequence,
            user_id=user_id,
            roles=list(key.roles),
            snapshot_id=snapshot,
            generation=self.state.generation,
            cache_hit=cache_hit,
            cache_key={
                "question": key.question,
                "user_id": key.user_id,
                "roles": list(key.roles),
                "snapshot_id": key.snapshot_id,
                "generation": key.generation,
                "policy_version": key.policy_version,
            },
            actions=copy.deepcopy(result.actions),
            evidence_ids=list(result.evidence_ids),
            source_families=list(result.source_families),
            rejected_unauthorized_ids=list(result.rejected_unauthorized_ids),
            rejected_stale_ids=list(result.rejected_stale_ids),
            rejected_untrusted_ids=list(result.rejected_untrusted_ids),
            stop_reason=result.stop_reason,
            action_count=len(result.actions),
            integrated_latency_ms=result.latency_ms,
            serving_latency_ms=elapsed,
            synthetic_tool_cost=self._cost(result, cache_hit),
        )
        return ServingResponse(result=result, trace=trace)

    def apply_frozen_g1_mutation(self) -> dict[str, Any]:
        started = time.perf_counter()
        before = self.state.generation
        self.state.apply_g1_mutation()
        return {
            "before_generation": before,
            "after_generation": self.state.generation,
            "active_snapshot": self.state.active_snapshot,
            "appended_event_ids": list(self.state.appended_event_ids),
            "replaced_shipment_ids": list(self.state.replaced_shipment_ids),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def load_scale_records(self, records: list[dict[str, Any]]) -> float:
        started = time.perf_counter()
        self.scale_records = {row["id"]: copy.deepcopy(row) for row in records}
        return (time.perf_counter() - started) * 1000

    def upsert_scale_record(self, record: dict[str, Any]) -> float:
        started = time.perf_counter()
        self.scale_records[record["id"]] = copy.deepcopy(record)
        return (time.perf_counter() - started) * 1000

    def delete_scale_record(self, record_id: str) -> float:
        started = time.perf_counter()
        self.scale_records.pop(record_id, None)
        return (time.perf_counter() - started) * 1000

    @property
    def logical_record_count(self) -> int:
        base = sum(
            len(rows)
            for rows in (
                self.copilot.data.customers,
                self.copilot.data.orders,
                self.copilot.data.shipments,
                self.copilot.data.events,
                self.copilot.data.invoices,
                self.copilot.data.finance,
                self.copilot.data.inventory,
                self.copilot.data.documents,
            )
        )
        return base + len(self.scale_records)

    def trace_dict(self, response: ServingResponse) -> dict[str, Any]:
        return asdict(response.trace)
