from pathlib import Path

from rag_practice.evaluation.otc import evaluate_baselines
from rag_practice.otc.baselines import BaselineSuite
from rag_practice.otc.data import OtcData


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
