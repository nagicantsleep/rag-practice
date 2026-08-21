from __future__ import annotations

import json
from pathlib import Path

from rag_practice.evaluation.metadata_filter import evaluate_filter_system
from rag_practice.metadata_filter import FilterAwareBM25, FilterRequest
from rag_practice.sources.base import SourceRecord

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "m08_metadata"
RESULTS = Path(__file__).resolve().parent / "results"


def load_records() -> dict[str, SourceRecord]:
    records = {}
    for line in (BENCH / "records.jsonl").read_text().splitlines():
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


def load_cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (BENCH / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    records = load_records()
    cases = load_cases()
    case_map = {str(case["id"]): case for case in cases}
    retriever = FilterAwareBM25(records)
    configs = {
        "unfiltered_bm25": ("unfiltered", None),
        "postfilter_k2": ("postfilter", 2),
        "postfilter_oversample_k5": ("postfilter", 5),
        "prefilter_bm25": ("prefilter", None),
    }

    systems = {}
    per_query = {}
    for system_name, (strategy, candidate_limit) in configs.items():
        traces = {}
        rows = []
        for case in cases:
            qid = str(case["id"])
            request = FilterRequest(**case["filters"])
            trace = retriever.search(
                str(case["query"]),
                request,
                strategy=strategy,
                limit=3,
                candidate_limit=candidate_limit,
            )
            traces[qid] = trace
            ids = [hit.record.id for hit in trace.hits]
            top = trace.hits[0].record if trace.hits else None
            reference = str(case.get("reference", ""))
            answer_correct = (
                reference.lower() in top.content.lower()
                if top is not None and reference
                else bool(case.get("expect_empty") and top is None)
            )
            rows.append(
                {
                    "id": qid,
                    "task": case["task"],
                    "query": case["query"],
                    "filters": case["filters"],
                    "relevant": case["relevant"],
                    "ranking": ids,
                    "top_locator": top.locator if top else "",
                    "top_metadata": dict(top.metadata) if top else {},
                    "answer_correct": answer_correct,
                    "records_indexed_for_lexical_search": trace.records_indexed_for_lexical_search,
                    "ranked_candidates_examined": trace.ranked_candidates_examined,
                    "rejected_after_ranking": trace.rejected_after_ranking,
                    "eligible_records": trace.eligible_records,
                    "latency_ms": trace.latency_ms,
                }
            )
        metrics = evaluate_filter_system(traces, case_map, records)
        systems[system_name] = {"metrics": metrics}
        per_query[system_name] = rows

    result = {
        "hypothesis": (
            "filter placement is part of retrieval correctness: unfiltered ranking can leak "
            "unauthorized records, while post-filtering can be safe yet lose recall when "
            "forbidden or mismatched records consume the candidate budget"
        ),
        "benchmark": {
            "records": len(records),
            "queries": len(cases),
            "tenants": ["alpha", "beta", "shared"],
            "hard_security_constraints": ["tenant", "role"],
            "explicit_query_filters": ["product", "region", "updated_after", "updated_before"],
        },
        "systems": systems,
        "per_query": per_query,
        "guardrails": [
            "tenant/role are treated as hard authorization predicates, not relevance features",
            "qrels/reference strings are used only after retrieval",
            "BM25 text is identical across systems; only filter placement/candidate budget changes",
            "postfilter oversampling is a cost/recall control, not a security substitute for prefiltering",
            "grounded extractive text can still be unauthorized or filter-invalid",
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# M08.3 Metadata / Filter-aware RAG results",
        "",
        f"Benchmark: {len(records)} records, {len(cases)} queries.",
        "",
        "| System | Recall@3 | Hit@1 | Constraint satisfied | Security leakage | Filter violation | Empty correct | Answer correct | Indexed records | Examined candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in configs:
        m = systems[name]["metrics"]
        lines.append(
            f"| {name} | {m['recall@3']:.3f} | {m['hit_rate@1']:.3f} | "
            f"{m['constraint_satisfaction_rate']:.3f} | {m['security_leakage_rate']:.3f} | "
            f"{m['explicit_filter_violation_rate']:.3f} | {m['empty_filter_accuracy']:.3f} | "
            f"{m['answer_correct_rate']:.3f} | {m['mean_records_indexed_for_lexical_search']:.1f} | "
            f"{m['mean_ranked_candidates_examined']:.1f} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Unfiltered Recall can look excellent while returning unauthorized or filter-invalid records.",
        "- Post-filtering removes leaks from returned results, but a small pre-filter candidate budget can destroy recall.",
        "- Oversampling can recover post-filter recall at higher candidate cost; it does not make hard authorization a relevance problem.",
        "- Pre-filtering applies hard predicates before lexical ranking and is the security-oriented candidate path.",
        "- This is a tiny deterministic corpus; latency/candidate counts illustrate mechanisms rather than production serving performance.",
    ]
    (RESULTS / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
