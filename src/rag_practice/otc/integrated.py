"""M11.2 bounded integrated Order-to-Cash/logistics copilot.

The runtime is deliberately qrel-blind: it reads only OtcData runtime sources and
uses the frozen M11.2 control rules. benchmark.json is never loaded here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .data import OtcData, Snapshot


_ORDER_RE = re.compile(r"\bSO-[A-Z0-9]+\b", re.IGNORECASE)
_HOURS_RE = re.compile(r"\b(\d+)\s+hours?\b", re.IGNORECASE)
_PROGRESS_CODES = {"PICKED_UP", "LINEHAUL", "OUT_FOR_DELIVERY", "ARRIVED_CUSTOMS"}
_POLICY_BY_EXCEPTION = {
    "WX_HOLD": "SOP-WEATHER",
    "CUSTOMS_REVIEW": "SOP-CUSTOMS",
    "DELAY_NOTICE": "SOP-UNKNOWN",
    "ADDR_CHECK": "SOP-ADDRESS",
    "VEHICLE_BREAKDOWN": "SOP-MECHANICAL",
}
_ACTION_BY_POLICY = {
    "SOP-WEATHER": "OPEN_CONTROL_TOWER_AND_NOTIFY_CUSTOMER",
    "SOP-CUSTOMS": "VERIFY_BROKER_CHECKLIST_AND_MONITOR",
    "SOP-UNKNOWN": "REQUEST_CARRIER_ROOT_CAUSE",
    "SOP-ADDRESS": "CONFIRM_ADDRESS_WITH_MASTER_AND_CARRIER",
    "SOP-MECHANICAL": "CARRIER_RECOVERY_AND_RECALCULATE_ETA",
}


@dataclass
class IntegratedResult:
    system: str = "integrated_copilot"
    answer: dict[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    source_families: list[str] = field(default_factory=list)
    retrieved_documents: list[tuple[str, float]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    rejected_unauthorized_ids: list[str] = field(default_factory=list)
    rejected_stale_ids: list[str] = field(default_factory=list)
    rejected_untrusted_ids: list[str] = field(default_factory=list)
    stop_reason: str = ""
    latency_ms: float = 0.0


class IntegratedCopilot:
    """Deterministic bounded investigation loop frozen by M11_2_CONTROL.md."""

    def __init__(self, data_root: str | Path) -> None:
        self.data = OtcData(data_root)
        self.contracts_by_customer: dict[str, list[dict[str, Any]]] = {}
        for row in self.data.contracts:
            self.contracts_by_customer.setdefault(row["customer_id"], []).append(row)
        self.policy_by_id = {row["id"]: row for row in self.data.policies}

    def _order_id(self, question: str) -> str | None:
        match = _ORDER_RE.search(question)
        return match.group(0).upper() if match else None

    def _customer(self, question: str, snapshot: Snapshot) -> dict[str, Any] | None:
        order_id = self._order_id(question)
        if order_id and order_id in snapshot.orders:
            return snapshot.customers[snapshot.orders[order_id]["customer_id"]]
        lowered = question.casefold()
        for row in snapshot.customers.values():
            if row["name"].casefold() in lowered:
                return row
        return None

    @staticmethod
    def _event_base_shipment_id(shipment: dict[str, Any]) -> str:
        return str(shipment["id"]).split("@")[0]

    def _order_context(self, order_id: str, snapshot: Snapshot) -> dict[str, Any] | None:
        order = snapshot.orders.get(order_id)
        if not order:
            return None
        shipment = snapshot.shipments.get(order["shipment_id"])
        invoice = snapshot.invoices.get(order["invoice_id"])
        inventory = next(
            (row for row in snapshot.inventory.values() if row["order_id"] == order_id),
            None,
        )
        base_shipment = self._event_base_shipment_id(shipment) if shipment else ""
        events = sorted(
            (row for row in snapshot.events.values() if row["shipment_id"] == base_shipment),
            key=lambda row: row["ts"],
        )
        current_event = None
        if shipment and shipment.get("latest_event_id"):
            current_event = snapshot.events.get(shipment["latest_event_id"])
        return {
            "order": order,
            "customer": snapshot.customers[order["customer_id"]],
            "shipment": shipment,
            "invoice": invoice,
            "inventory": inventory,
            "events": events,
            "current_event": current_event,
        }

    def _finance_context(self, order_id: str, user_id: str, snapshot: Snapshot) -> tuple[dict[str, Any] | None, bool]:
        if not self.data.can_read("finance", user_id):
            return None, False
        order = snapshot.orders.get(order_id)
        if not order:
            return None, True
        row = next(
            (item for item in snapshot.finance.values() if item["invoice_id"] == order["invoice_id"]),
            None,
        )
        return row, True

    @staticmethod
    def _effective(row: dict[str, Any], as_of: str) -> bool:
        day = datetime.fromisoformat(as_of.replace("Z", "+00:00")).date()
        start = date.fromisoformat(row["effective_from"])
        end = date.fromisoformat(row["effective_to"]) if row.get("effective_to") else None
        return start <= day and (end is None or day <= end)

    def _active_contract(self, customer_id: str, snapshot: Snapshot) -> tuple[dict[str, Any] | None, list[str], list[str]]:
        active: list[dict[str, Any]] = []
        stale: list[str] = []
        untrusted: list[str] = []
        for row in self.contracts_by_customer.get(customer_id, []):
            if not row.get("trusted", False):
                untrusted.append(row["id"])
                continue
            if not self._effective(row, snapshot.as_of):
                stale.append(row["id"])
                continue
            active.append(row)
        active.sort(key=lambda row: (row["effective_from"], row["version"]), reverse=True)
        return (active[0] if active else None), stale, untrusted

    def _policy(self, exception_code: str, user_id: str) -> tuple[dict[str, Any] | None, list[str], list[str]]:
        policy_id = _POLICY_BY_EXCEPTION.get(exception_code)
        if not policy_id:
            return None, [], []
        row = self.policy_by_id.get(policy_id)
        if not row:
            return None, [], []
        if not row.get("trusted", False):
            return None, [], [row["id"]]
        if not (self.data.roles(user_id) & set(row.get("allowed_roles", []))):
            return None, [row["id"]], []
        return row, [], []

    @staticmethod
    def _hours(contract: dict[str, Any] | None) -> int | None:
        if not contract:
            return None
        match = _HOURS_RE.search(contract["text"])
        return int(match.group(1)) if match else None

    @staticmethod
    def _sla_breached(contract: dict[str, Any] | None, events: list[dict[str, Any]], as_of: str) -> tuple[bool | None, int | None]:
        hours = IntegratedCopilot._hours(contract)
        pickup = next((row for row in events if row["code"] == "PICKED_UP" and row.get("trusted", False)), None)
        if hours is None or pickup is None:
            return None, hours
        start = datetime.fromisoformat(pickup["ts"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        return (end - start).total_seconds() > hours * 3600, hours

    @staticmethod
    def _add_evidence(result: IntegratedResult, row: dict[str, Any] | None, family: str) -> None:
        if not row:
            return
        row_id = row["id"]
        if row_id not in result.evidence_ids:
            result.evidence_ids.append(row_id)
        if family not in result.source_families:
            result.source_families.append(family)

    def run(self, question: str, user_id: str, snapshot_id: str) -> IntegratedResult:
        started = time.perf_counter()
        snapshot = self.data.snapshot(snapshot_id)
        result = IntegratedResult()
        lowered = question.casefold()
        order_id = self._order_id(question)
        customer = self._customer(question, snapshot)

        asks_finance = any(term in lowered for term in ("payment", "credit", "finance", "hold reason"))
        asks_contract = any(term in lowered for term in ("sla", "breach", "commitment", "contract"))
        asks_status = any(term in lowered for term in ("status", "eta", "state", "delivered", "transit"))
        asks_cause = any(term in lowered for term in ("why", "cause", "exception", "delay", "investigate"))
        asks_action = any(term in lowered for term in ("action", "procedure", "escalat", "investigate"))
        asks_inventory = any(term in lowered for term in ("inventory", "fulfillment", "not dispatched", "blocker"))
        asks_invoice = "invoice" in lowered

        context: dict[str, Any] | None = None
        if order_id:
            context = self._order_context(order_id, snapshot)
            result.actions.append({"action": "order_context", "argument": order_id, "found": context is not None})
            if context:
                self._add_evidence(result, context["order"], "erp_order")
                if context["shipment"] and (asks_status or asks_cause or asks_contract or asks_action):
                    self._add_evidence(result, context["shipment"], "logistics")
                if asks_invoice or asks_finance:
                    self._add_evidence(result, context["invoice"], "erp_order")
                if asks_inventory:
                    self._add_evidence(result, context["inventory"], "inventory")
                if asks_status or asks_cause or asks_contract or asks_action:
                    current = context["current_event"]
                    self._add_evidence(result, current, "logistics")
                    if asks_contract:
                        pickup = next((row for row in context["events"] if row["code"] == "PICKED_UP"), None)
                        self._add_evidence(result, pickup, "logistics")
                    delivered = next((row for row in context["events"] if row["code"] == "DELIVERED"), None)
                    if delivered and context["shipment"].get("status") != "DELIVERED" and any(term in lowered for term in ("delivered", "transit", "status")):
                        self._add_evidence(result, delivered, "logistics")

        if asks_finance and order_id:
            finance, authorized = self._finance_context(order_id, user_id, snapshot)
            result.actions.append({"action": "finance_context", "argument": order_id, "authorized": authorized, "found": finance is not None})
            if not authorized:
                result.evidence_ids.append("AUTH-FIN")
                result.source_families.append("authorization")
                result.rejected_unauthorized_ids.append("finance")
                result.answer = {"decision": "DENIED"}
                result.stop_reason = "authorization_denied"
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result
            result.evidence_ids.append("AUTH-FIN")
            if "authorization" not in result.source_families:
                result.source_families.append("authorization")
            self._add_evidence(result, finance, "finance")
            if finance:
                result.answer.update(
                    payment_status=finance["payment_status"],
                    credit_hold=finance["credit_hold"],
                    hold_reason=finance["hold_reason"],
                )
                if "investigate" in lowered or "blocker" in lowered:
                    result.answer["finance_blocker"] = bool(finance["credit_hold"])

        contract = None
        if asks_contract and customer and len(result.actions) < 4:
            contract, stale_ids, untrusted_ids = self._active_contract(customer["id"], snapshot)
            result.rejected_stale_ids.extend(stale_ids)
            result.rejected_untrusted_ids.extend(untrusted_ids)
            result.actions.append({"action": "active_contract", "argument": customer["id"], "selected": contract["id"] if contract else None})
            self._add_evidence(result, contract, "contracts")
            if contract:
                hours = self._hours(contract)
                if "delivery commitment" in lowered or "commitment" in lowered:
                    result.answer["delivery_commitment_hours"] = hours
                if "using the contract effective" in lowered:
                    result.answer["contract_id"] = contract["id"]
                    result.answer["delivery_commitment_hours"] = hours
                if "weather" in contract["text"].casefold() and "waiver" in lowered:
                    result.answer["weather_waiver"] = False
                if context and ("sla" in lowered or "breach" in lowered):
                    breached, hours = self._sla_breached(contract, context["events"], snapshot.as_of)
                    result.answer["sla_breached"] = breached
                    result.answer["commitment_hours"] = hours

        if context:
            shipment = context["shipment"]
            current = context["current_event"]
            events = context["events"]
            invoice = context["invoice"]
            inventory = context["inventory"]

            if asks_invoice and invoice:
                result.answer.update(
                    invoice_id=invoice["id"],
                    amount_jpy=invoice["amount_jpy"],
                    invoice_status=invoice["status"],
                )

            if shipment and ("current eta" in lowered or ("status" in lowered and "eta" in lowered)):
                result.answer["shipment_status"] = shipment["status"]
                result.answer["eta"] = shipment["eta"]
                result.answer["latest_event_code"] = current["code"] if current else None

            if shipment and "current" in lowered and "shipment" in lowered and "state" in lowered:
                code = current["code"] if current else None
                result.answer.update(
                    shipment_status=shipment["status"],
                    latest_event_code=code,
                    confirmed_exception=bool(code and code not in _PROGRESS_CODES),
                )

            delivered = next((row for row in events if row["code"] == "DELIVERED" and row.get("trusted", False)), None)
            if shipment and delivered and shipment["status"] != "DELIVERED" and any(term in lowered for term in ("delivered", "transit")):
                result.answer.update(
                    decision="CONFLICT",
                    erp_status=shipment["status"],
                    carrier_status="DELIVERED",
                )
                result.stop_reason = "authoritative_source_conflict"

            if inventory and asks_inventory and inventory["shortage"]:
                result.answer.update(
                    blocker="INVENTORY_SHORTAGE",
                    backorder_qty=inventory["backorder_qty"],
                )
                if "credit" in lowered and not asks_finance:
                    result.answer["credit_hold"] = "NOT_READ"

            code = current["code"] if current else None
            confirmed_exception = bool(code and code not in _PROGRESS_CODES)
            if asks_cause:
                if code == "DELAY_NOTICE" or not confirmed_exception:
                    result.answer["root_cause"] = "UNKNOWN"
                else:
                    result.answer["root_cause"] = code
                if "confirmed exception" in lowered:
                    result.answer["confirmed_exception"] = confirmed_exception

            if asks_action and len(result.actions) < 4:
                policy_code = code if confirmed_exception else None
                if code == "DELAY_NOTICE":
                    policy_code = "DELAY_NOTICE"
                if policy_code:
                    policy, denied_ids, untrusted_ids = self._policy(policy_code, user_id)
                    result.rejected_unauthorized_ids.extend(denied_ids)
                    result.rejected_untrusted_ids.extend(untrusted_ids)
                    result.actions.append({"action": "policy_search", "argument": policy_code, "selected": policy["id"] if policy else None})
                    self._add_evidence(result, policy, "sop")
                    if policy:
                        result.retrieved_documents.append((policy["id"], 1.0))
                        result.answer["recommended_action"] = _ACTION_BY_POLICY[policy["id"]]

        # Explicitly account for the retained untrusted corpus without exposing it.
        if asks_action or asks_cause:
            for row in self.data.untrusted:
                if row["id"] not in result.rejected_untrusted_ids:
                    result.rejected_untrusted_ids.append(row["id"])

        result.source_families.sort()
        if not result.stop_reason:
            if result.answer.get("root_cause") == "UNKNOWN":
                result.stop_reason = "insufficient_confirmed_cause"
            elif result.answer.get("decision") == "CONFLICT":
                result.stop_reason = "authoritative_source_conflict"
            elif result.answer:
                result.stop_reason = "requested_fields_supported"
            else:
                result.answer = {"root_cause": "UNKNOWN"}
                result.stop_reason = "insufficient_evidence"
        if len(result.actions) > 4:
            raise AssertionError("M11.2 source-action budget exceeded")
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result
