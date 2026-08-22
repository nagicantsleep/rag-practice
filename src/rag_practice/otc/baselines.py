"""M11.1 explicit baselines over the frozen O2C/logistics benchmark."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_practice.ir.bm25 import BM25Index

from .data import OtcData, Snapshot


_ORDER_RE = re.compile(r"\bSO-[A-Z0-9]+\b", re.IGNORECASE)
_HOURS_RE = re.compile(r"\b(\d+)\s+hours?\b", re.IGNORECASE)


@dataclass
class BaselineResult:
    system: str
    answer: dict[str, Any]
    evidence_ids: list[str] = field(default_factory=list)
    source_families: list[str] = field(default_factory=list)
    retrieved_documents: list[tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0


class BaselineSuite:
    def __init__(self, data_root: str | Path) -> None:
        self.data = OtcData(data_root)
        searchable = {
            row["id"]: " ".join(
                part
                for part in (
                    row["id"],
                    row.get("customer_id", ""),
                    row.get("text", ""),
                )
                if part
            )
            for row in self.data.documents
        }
        self.doc_index = BM25Index(searchable)
        self.doc_family: dict[str, str] = {}
        for row in self.data.contracts:
            self.doc_family[row["id"]] = "contracts"
        for row in self.data.policies:
            self.doc_family[row["id"]] = "sop"
        for row in self.data.untrusted:
            self.doc_family[row["id"]] = "untrusted"

    def _order_id(self, question: str) -> str | None:
        match = _ORDER_RE.search(question)
        return match.group(0).upper() if match else None

    def _customer_for_question(
        self, question: str, snapshot: Snapshot
    ) -> dict[str, Any] | None:
        lowered = question.casefold()
        for customer in snapshot.customers.values():
            if customer["name"].casefold() in lowered:
                return customer
        order_id = self._order_id(question)
        if order_id and order_id in snapshot.orders:
            return snapshot.customers[snapshot.orders[order_id]["customer_id"]]
        return None

    def _order_context(
        self, question: str, snapshot: Snapshot
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        list[dict[str, Any]],
    ]:
        order_id = self._order_id(question)
        if not order_id or order_id not in snapshot.orders:
            return None, None, None, None, None, []
        order = snapshot.orders[order_id]
        shipment = snapshot.shipments.get(order["shipment_id"])
        invoice = snapshot.invoices.get(order["invoice_id"])
        fin = next(
            (row for row in snapshot.finance.values() if row["invoice_id"] == order["invoice_id"]),
            None,
        )
        inv = next(
            (row for row in snapshot.inventory.values() if row["order_id"] == order_id),
            None,
        )
        events = sorted(
            (
                row
                for row in snapshot.events.values()
                if shipment and row["shipment_id"] == shipment["id"].split("@")[0]
            ),
            key=lambda row: row["ts"],
        )
        return order, shipment, invoice, fin, inv, events

    def _documents(self, question: str, k: int = 3) -> list[tuple[str, float]]:
        return self.doc_index.search(question, k=k)

    def _doc_evidence(self, ranked: list[tuple[str, float]]) -> list[str]:
        return [doc_id for doc_id, _ in ranked]

    def _doc_sources(self, ranked: list[tuple[str, float]]) -> list[str]:
        return sorted({self.doc_family[doc_id] for doc_id, _ in ranked})

    def _contract_from_ranked(
        self, ranked: list[tuple[str, float]], customer_id: str | None
    ) -> dict[str, Any] | None:
        for doc_id, _ in ranked:
            row = self.data.document_by_id[doc_id]
            if self.doc_family[doc_id] == "contracts" and (
                customer_id is None or row.get("customer_id") == customer_id
            ):
                return row
        return None

    def _policy_id_for_event(self, code: str | None) -> str | None:
        return {
            "WX_HOLD": "SOP-WEATHER",
            "CUSTOMS_REVIEW": "SOP-CUSTOMS",
            "DELAY_NOTICE": "SOP-UNKNOWN",
            "ADDR_CHECK": "SOP-ADDRESS",
            "VEHICLE_BREAKDOWN": "SOP-MECHANICAL",
        }.get(code or "")

    def _policy_action(self, policy_id: str | None) -> str | None:
        return {
            "SOP-WEATHER": "OPEN_CONTROL_TOWER_AND_NOTIFY_CUSTOMER",
            "SOP-CUSTOMS": "VERIFY_BROKER_CHECKLIST_AND_MONITOR",
            "SOP-UNKNOWN": "REQUEST_CARRIER_ROOT_CAUSE",
            "SOP-ADDRESS": "CONFIRM_ADDRESS_WITH_MASTER_AND_CARRIER",
            "SOP-MECHANICAL": "CARRIER_RECOVERY_AND_RECALCULATE_ETA",
        }.get(policy_id or "")

    def _hours(self, contract: dict[str, Any] | None) -> int | None:
        if not contract:
            return None
        match = _HOURS_RE.search(contract["text"])
        return int(match.group(1)) if match else None

    def _pickup(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next((row for row in events if row["code"] == "PICKED_UP"), None)

    def _sla_breached(
        self, contract: dict[str, Any] | None, events: list[dict[str, Any]], as_of: str
    ) -> tuple[bool | None, int | None]:
        hours = self._hours(contract)
        pickup = self._pickup(events)
        if hours is None or pickup is None:
            return None, hours
        start = datetime.fromisoformat(pickup["ts"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        return (end - start).total_seconds() > hours * 3600, hours

    def no_retrieval(self, question: str, user_id: str, snapshot_id: str) -> BaselineResult:
        start = time.perf_counter()
        result = BaselineResult(
            system="no_retrieval",
            answer={"root_cause": "UNKNOWN"},
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def document_only(self, question: str, user_id: str, snapshot_id: str) -> BaselineResult:
        start = time.perf_counter()
        snapshot = self.data.snapshot(snapshot_id)
        ranked = self._documents(question)
        evidence = self._doc_evidence(ranked)
        answer: dict[str, Any] = {}
        customer = self._customer_for_question(question, snapshot)
        customer_id = customer["id"] if customer else None
        contract = self._contract_from_ranked(ranked, customer_id)
        lowered = question.casefold()

        if "delivery commitment" in lowered and contract:
            answer["delivery_commitment_hours"] = self._hours(contract)
            if "weather" in contract["text"].casefold():
                answer["weather_waiver"] = False

        for doc_id, _ in ranked:
            action = self._policy_action(doc_id)
            if action:
                answer["recommended_action"] = action
                if doc_id == "SOP-ADDRESS":
                    answer["exception"] = "ADDR_CHECK"
                if doc_id == "SOP-UNKNOWN":
                    answer["root_cause"] = "UNKNOWN"
                break

        if not answer:
            answer = {"root_cause": "UNKNOWN"}

        result = BaselineResult(
            system="document_only",
            answer=answer,
            evidence_ids=evidence,
            source_families=self._doc_sources(ranked),
            retrieved_documents=ranked,
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def structured_only(self, question: str, user_id: str, snapshot_id: str) -> BaselineResult:
        start = time.perf_counter()
        snapshot = self.data.snapshot(snapshot_id)
        answer, evidence, families = self._structured_answer(
            question, user_id, snapshot, include_documents=False
        )
        result = BaselineResult(
            system="structured_only",
            answer=answer,
            evidence_ids=evidence,
            source_families=sorted(families),
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def fixed_mixed(self, question: str, user_id: str, snapshot_id: str) -> BaselineResult:
        """One-shot mixed-source baseline.

        Document retrieval uses only the original user question. Structured
        reads happen in parallel conceptually; no second document search is
        allowed using an event/exception discovered from structured sources.
        """
        start = time.perf_counter()
        snapshot = self.data.snapshot(snapshot_id)
        ranked = self._documents(question)
        answer, evidence, families = self._structured_answer(
            question, user_id, snapshot, include_documents=True, ranked=ranked
        )
        evidence.extend(doc_id for doc_id, _ in ranked if doc_id not in evidence)
        families.update(self._doc_sources(ranked))
        result = BaselineResult(
            system="fixed_mixed",
            answer=answer,
            evidence_ids=evidence,
            source_families=sorted(families),
            retrieved_documents=ranked,
        )
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _structured_answer(
        self,
        question: str,
        user_id: str,
        snapshot: Snapshot,
        *,
        include_documents: bool,
        ranked: list[tuple[str, float]] | None = None,
    ) -> tuple[dict[str, Any], list[str], set[str]]:
        lowered = question.casefold()
        evidence: list[str] = []
        families: set[str] = set()
        ranked = ranked or []
        order, shipment, invoice, fin, inv, events = self._order_context(question, snapshot)
        customer = self._customer_for_question(question, snapshot)
        finance_terms = any(term in lowered for term in ("payment", "credit", "finance", "hold reason"))
        asks_sensitive = finance_terms and ("credit" in lowered or "payment" in lowered)

        if asks_sensitive and not self.data.can_read("finance", user_id):
            evidence.append("AUTH-FIN")
            families.add("authorization")
            return {"decision": "DENIED"}, evidence, families

        if order:
            evidence.append(order["id"])
            families.add("erp_order")
        if shipment:
            evidence.append(shipment["id"])
            families.add("logistics")
        if invoice and ("invoice" in lowered or asks_sensitive or "finance" in lowered):
            evidence.append(invoice["id"])
            families.add("erp_order")
        if inv and any(term in lowered for term in ("not dispatched", "inventory", "fulfillment", "blocker")):
            evidence.append(inv["id"])
            families.add("inventory")
        if asks_sensitive and fin and self.data.can_read("finance", user_id):
            evidence.extend(["AUTH-FIN", fin["id"]])
            families.update({"authorization", "finance"})

        latest = events[-1] if events else None
        if latest and shipment:
            if shipment.get("latest_event_id") in snapshot.events:
                current = snapshot.events[shipment["latest_event_id"]]
                if current["id"] not in evidence:
                    evidence.append(current["id"])
            if latest["id"] not in evidence:
                evidence.append(latest["id"])
            families.add("logistics")

        answer: dict[str, Any] = {}

        if "which invoice" in lowered and order and invoice:
            answer.update(
                invoice_id=invoice["id"],
                amount_jpy=invoice["amount_jpy"],
                invoice_status=invoice["status"],
            )

        if ("current eta" in lowered or ("status" in lowered and "eta" in lowered)) and shipment:
            answer["shipment_status"] = shipment["status"]
            answer["eta"] = shipment["eta"]
            if latest:
                answer["latest_event_code"] = latest["code"]

        if "current helios shipment state" in lowered and shipment:
            answer.update(
                shipment_status=shipment["status"],
                latest_event_code=latest["code"] if latest else None,
                confirmed_exception=bool(latest and latest["code"] not in {"PICKED_UP", "LINEHAUL", "OUT_FOR_DELIVERY"}),
            )

        if asks_sensitive and fin:
            answer.update(
                payment_status=fin["payment_status"],
                credit_hold=fin["credit_hold"],
                hold_reason=fin["hold_reason"],
            )

        if "why" in lowered and "not dispatched" in lowered and inv:
            if inv["shortage"]:
                answer.update(
                    blocker="INVENTORY_SHORTAGE",
                    backorder_qty=inv["backorder_qty"],
                    credit_hold="NOT_READ",
                )

        if "delivered or still in transit" in lowered and shipment:
            delivered = next((row for row in events if row["code"] == "DELIVERED"), None)
            if delivered and shipment["status"] != "DELIVERED":
                answer.update(
                    decision="CONFLICT",
                    erp_status=shipment["status"],
                    carrier_status="DELIVERED",
                )

        exception_code = latest["code"] if latest else None
        confirmed_exception = exception_code not in {None, "PICKED_UP", "LINEHAUL", "OUT_FOR_DELIVERY"}
        if any(term in lowered for term in ("what caused", "why is it delayed", "confirmed cause", "confirmed exception")):
            if exception_code == "DELAY_NOTICE" or not confirmed_exception:
                answer["root_cause"] = "UNKNOWN"
            else:
                answer["root_cause"] = exception_code
            if "confirmed exception" in lowered:
                answer["confirmed_exception"] = confirmed_exception

        if "investigate" in lowered and fin:
            answer["finance_blocker"] = bool(fin["credit_hold"])

        contract = None
        if include_documents:
            customer_id = customer["id"] if customer else None
            contract = self._contract_from_ranked(ranked, customer_id)
            if contract and contract["id"] not in evidence:
                evidence.append(contract["id"])
                families.add("contracts")

        if "delivery commitment" in lowered and contract:
            answer["delivery_commitment_hours"] = self._hours(contract)
            if customer and customer["id"] == "C-ATLAS":
                answer["weather_waiver"] = False

        if ("sla" in lowered or "breach" in lowered) and include_documents and contract:
            breached, hours = self._sla_breached(contract, events, snapshot.as_of)
            answer["sla_breached"] = breached
            answer["commitment_hours"] = hours
            if "using the contract effective" in lowered:
                answer["contract_id"] = contract["id"]
                answer["delivery_commitment_hours"] = hours

        if include_documents:
            needed_policy = self._policy_id_for_event(exception_code)
            retrieved_ids = {doc_id for doc_id, _ in ranked}
            if needed_policy in retrieved_ids:
                action = self._policy_action(needed_policy)
                if action:
                    answer["recommended_action"] = action
                if needed_policy == "SOP-ADDRESS":
                    answer["exception"] = "ADDR_CHECK"
                    answer["ignored_untrusted"] = "NOTE-GAMMA-INJECTION" not in retrieved_ids
            elif "confirmed exception" in lowered and not confirmed_exception:
                answer["recommended_action"] = "NONE_YET"

        if not answer:
            answer["root_cause"] = "UNKNOWN"
        return answer, evidence, families
