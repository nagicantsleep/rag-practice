"""Frozen M11 O2C/logistics dataset loader."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Snapshot:
    id: str
    as_of: str
    customers: dict[str, dict[str, Any]]
    orders: dict[str, dict[str, Any]]
    shipments: dict[str, dict[str, Any]]
    events: dict[str, dict[str, Any]]
    invoices: dict[str, dict[str, Any]]
    finance: dict[str, dict[str, Any]]
    inventory: dict[str, dict[str, Any]]


class OtcData:
    """Runtime-safe view of M11 sources.

    This loader never reads benchmark.json. Evaluator labels remain outside the
    runtime source boundary.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = _load_json(self.root / "manifest.json")
        self.customers = _load_json(self.root / "structured/customers.json")
        self.orders = _load_json(self.root / "structured/orders.json")
        self.shipments = _load_json(self.root / "structured/shipments.json")
        self.events = _load_json(self.root / "structured/tracking_events.json")
        self.invoices = _load_json(self.root / "structured/invoices.json")
        self.finance = _load_json(self.root / "structured/finance.json")
        self.inventory = _load_json(self.root / "structured/inventory.json")
        self.auth = _load_json(self.root / "structured/auth.json")
        self.contracts = _load_json(self.root / "documents/contracts.json")
        self.policies = _load_json(self.root / "documents/policies.json")
        self.untrusted = _load_json(self.root / "documents/untrusted.json")
        self.mutations = _load_json(self.root / "mutations.json")

        self.documents = self.contracts + self.policies + self.untrusted
        self.document_by_id = {row["id"]: row for row in self.documents}
        self.user_by_id = {row["id"]: row for row in self.auth["users"]}
        self.auth_policy_by_family = {
            row["source_family"]: row for row in self.auth["policies"]
        }

    def roles(self, user_id: str) -> set[str]:
        user = self.user_by_id.get(user_id)
        return set(user["roles"]) if user else set()

    def can_read(self, source_family: str, user_id: str) -> bool:
        policy = self.auth_policy_by_family.get(source_family)
        if policy is None:
            return False
        return bool(self.roles(user_id) & set(policy["allowed_roles"]))

    def snapshot(self, snapshot_id: str) -> Snapshot:
        snapshot_meta = next(
            row for row in self.mutations["snapshots"] if row["id"] == snapshot_id
        )
        shipments = {row["id"]: copy.deepcopy(row) for row in self.shipments}
        events = {row["id"]: copy.deepcopy(row) for row in self.events}

        if snapshot_id != self.mutations["base_snapshot"]:
            for op in self.mutations["operations"]:
                if op["to_snapshot"] != snapshot_id:
                    continue
                record = copy.deepcopy(op["record"])
                if op["type"] == "append_tracking_event":
                    events[record["id"]] = record
                elif op["type"] == "replace_shipment_version":
                    base_id = record["base_id"]
                    versioned = {k: v for k, v in record.items() if k != "base_id"}
                    shipments[base_id] = versioned
                    shipments[record["id"]] = copy.deepcopy(record)
                else:
                    raise ValueError(f"unsupported mutation type: {op['type']}")

        return Snapshot(
            id=snapshot_id,
            as_of=snapshot_meta["as_of"],
            customers={row["id"]: row for row in self.customers},
            orders={row["id"]: row for row in self.orders},
            shipments=shipments,
            events=events,
            invoices={row["id"]: row for row in self.invoices},
            finance={row["id"]: row for row in self.finance},
            inventory={row["id"]: row for row in self.inventory},
        )
