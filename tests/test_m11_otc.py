from pathlib import Path

from rag_practice.evaluation.otc import evaluate_baselines
from rag_practice.evaluation.otc_integrated import evaluate_integrated
from rag_practice.otc.baselines import BaselineSuite
from rag_practice.otc.data import OtcData
from rag_practice.otc.integrated import IntegratedCopilot


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "m11_otc_logistics"


def test_m11_dataset_snapshot_mutation_is_versioned() -> None:
    data = OtcData(DATA)
    g0 = data.snapshot("g0")
    g1 = data.snapshot("g1")
    assert g0.shipments["SH-1008"]["status"] == "IN_TRANSIT"
    assert "EV-H003" not in g0.events
    assert g1.shipments["SH-1008"]["id"] == "SH-1008@g1"
    assert g1.shipments["SH-1008"]["status"] == "EXCEPTION"
    assert g1.events["EV-H003"]["code"] == "VEHICLE_BREAKDOWN"


def test_m11_finance_acl_is_explicit() -> None:
    data = OtcData(DATA)
    assert not data.can_read("finance", "U-OPS")
    assert data.can_read("finance", "U-FIN")
    assert data.can_read("finance", "U-MGR")


def test_m11_fixed_mixed_denies_sensitive_finance_before_read() -> None:
    suite = BaselineSuite(DATA)
    result = suite.fixed_mixed(
        "Show the payment status, credit-hold state, and hold reason for Cedar order SO-1003.",
        "U-OPS",
        "g0",
    )
    assert result.answer == {"decision": "DENIED"}
    assert "AUTH-FIN" in result.evidence_ids
    assert "FIN-1003" not in result.evidence_ids


def test_m11_naive_document_baseline_retains_stale_and_untrusted_failures() -> None:
    suite = BaselineSuite(DATA)
    stale = suite.document_only(
        "Using the contract effective at the benchmark time, what is Epsilon Retail's delivery commitment and is SO-1005 already in breach?",
        "U-OPS",
        "g0",
    )
    adversarial = suite.document_only(
        "What action should operations take for Gamma order SO-1007's address exception?",
        "U-OPS",
        "g0",
    )
    assert "CTR-EPS-v1" in stale.evidence_ids
    assert "NOTE-GAMMA-INJECTION" in adversarial.evidence_ids


def test_m11_evaluator_keeps_four_baselines_and_18_test_tasks() -> None:
    results = evaluate_baselines(DATA, split="test")
    assert set(results["systems"]) == {
        "no_retrieval",
        "document_only",
        "structured_only",
        "fixed_mixed",
    }
    for payload in results["systems"].values():
        assert len(payload["rows"]) == 18


def test_m11_integrated_finance_denial_fails_closed_before_sensitive_read() -> None:
    copilot = IntegratedCopilot(DATA)
    result = copilot.run(
        "Show the payment status, credit-hold state, and hold reason for Cedar order SO-1003.",
        "U-OPS",
        "g0",
    )
    assert result.answer == {"decision": "DENIED"}
    assert "AUTH-FIN" in result.evidence_ids
    assert "FIN-1003" not in result.evidence_ids
    assert result.stop_reason == "authorization_denied"
    assert len(result.actions) <= 4


def test_m11_integrated_uses_current_contract_without_stale_exposure() -> None:
    copilot = IntegratedCopilot(DATA)
    result = copilot.run(
        "Using the contract effective at the benchmark time, what is Epsilon Retail's delivery commitment and is SO-1005 already in breach?",
        "U-OPS",
        "g0",
    )
    assert "C-EPS" in result.evidence_ids
    assert "CTR-EPS-v2" in result.evidence_ids
    assert "CTR-EPS-v1" not in result.evidence_ids
    assert "CTR-EPS-v1" in result.rejected_stale_ids
    assert result.answer["delivery_commitment_hours"] == 72


def test_m11_integrated_targets_trusted_address_sop_and_rejects_injection() -> None:
    copilot = IntegratedCopilot(DATA)
    result = copilot.run(
        "What action should operations take for Gamma order SO-1007's address exception?",
        "U-OPS",
        "g0",
    )
    assert "SOP-ADDRESS" in result.evidence_ids
    assert "NOTE-GAMMA-INJECTION" not in result.evidence_ids
    assert "NOTE-GAMMA-INJECTION" in result.rejected_untrusted_ids
    assert result.answer["exception"] == "ADDR_CHECK"
    assert result.answer["recommended_action"] == "CONFIRM_ADDRESS_WITH_MASTER_AND_CARRIER"
    assert result.answer["ignored_untrusted"] is True


def test_m11_integrated_recognizes_natural_next_action_request() -> None:
    copilot = IntegratedCopilot(DATA)
    result = copilot.run(
        "Boreal order SO-1002 is in exception. What is the confirmed cause, is its SLA already breached, and what should operations do next?",
        "U-OPS",
        "g0",
    )
    assert "SOP-CUSTOMS" in result.evidence_ids
    assert result.answer["recommended_action"] == "VERIFY_BROKER_CHECKLIST_AND_MONITOR"


def test_m11_integrated_reports_finance_blocker_semantically() -> None:
    copilot = IntegratedCopilot(DATA)
    result = copilot.run(
        "Why is Cedar order SO-1003 on hold? Include the finance blocker and its recorded reason.",
        "U-FIN",
        "g0",
    )
    assert result.answer["blocker"] == "CREDIT_HOLD"
    assert result.answer["credit_hold"] is True
    assert result.answer["hold_reason"] == "Credit limit exceeded after overdue balance review."


def test_m11_integrated_keeps_inventory_blocker_independent_from_finance() -> None:
    copilot = IntegratedCopilot(DATA)
    result = copilot.run(
        "Why has Kappa order SO-1011 not dispatched?",
        "U-OPS",
        "g0",
    )
    assert result.answer["blocker"] == "INVENTORY_SHORTAGE"
    assert result.answer["backorder_qty"] == 24
    assert result.answer["credit_hold"] == "NOT_READ"
    assert not any(action["action"] == "finance_context" for action in result.actions)


def test_m11_integrated_does_not_speculate_escalation_without_exception() -> None:
    copilot = IntegratedCopilot(DATA)
    result = copilot.run(
        "At snapshot g0, what confirmed exception explains Helios order SO-1008 and what escalation applies?",
        "U-OPS",
        "g0",
    )
    assert result.answer["root_cause"] == "UNKNOWN"
    assert result.answer["confirmed_exception"] is False
    assert result.answer["recommended_action"] == "NONE_YET"
    assert not any(action["action"] == "policy_search" for action in result.actions)


def test_m11_integrated_respects_action_budget_and_evaluates_all_tasks() -> None:
    results = evaluate_integrated(DATA, split="test")
    assert len(results["rows"]) == 18
    assert results["metrics"]["max_action_count"] <= 4
    assert all(row["action_count"] <= 4 for row in results["rows"])
