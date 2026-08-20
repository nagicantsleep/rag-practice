from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.adaptive.control import AdaptiveRAGController
from rag_practice.adaptive.router import AlwaysSingleRouter, KeywordRouter, NaiveBayesRouteClassifier, Route
from rag_practice.evaluation.adaptive import evaluate_control_traces

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/m06_adaptive"
OUT = ROOT / "labs/06_adaptive_corrective/results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class OracleRouteRouter:
    def __init__(self, rows: list[dict]) -> None:
        self.routes = {row["query"]: Route(row["route"]) for row in rows}

    def route(self, query: str) -> Route:
        return self.routes[query]


def run_router(name: str, router, rows: list[dict], primary: dict[str, str], fallback: dict[str, str]) -> dict:
    controller = AdaptiveRAGController(
        router=router,
        primary_documents=primary,
        fallback_documents=fallback,
        max_iterative_steps=2,
    )
    traces = {}
    latencies = []
    for row in rows:
        start = perf_counter()
        trace = controller.run(row["query"])
        latencies.append((perf_counter() - start) * 1000)
        traces[row["id"]] = trace

    metrics = evaluate_control_traces(rows, traces)
    metrics["mean_control_ms"] = fmean(latencies)
    serial = {
        query_id: {
            "route": trace.route.value,
            "retrieval_calls": trace.retrieval_calls,
            "selected_document_ids": list(trace.selected_document_ids),
            "correction_triggered": trace.correction_triggered,
            "steps": [
                {
                    "query": step.query,
                    "source": step.source,
                    "document_id": step.document_id,
                    "score": step.score,
                    "assessment": step.assessment.value,
                }
                for step in trace.steps
            ],
        }
        for query_id, trace in traces.items()
    }
    return {"name": name, "metrics": metrics, "traces": serial}


def main() -> None:
    rows = load_jsonl(BENCH / "queries.jsonl")
    train = load_jsonl(BENCH / "route_train.jsonl")
    primary = {row["id"]: row["text"] for row in load_jsonl(BENCH / "documents.jsonl")}
    fallback = {row["id"]: row["text"] for row in load_jsonl(BENCH / "fallback_documents.jsonl")}

    learned = NaiveBayesRouteClassifier(alpha=1.0)
    learned.fit((row["query"], row["route"]) for row in train)

    systems = [
        run_router("always_single", AlwaysSingleRouter(), rows, primary, fallback),
        run_router("keyword_router", KeywordRouter(), rows, primary, fallback),
        run_router("naive_bayes_router", learned, rows, primary, fallback),
        run_router("oracle_route_ceiling", OracleRouteRouter(rows), rows, primary, fallback),
    ]

    payload = {
        "experiment_id": "m06_adaptive_control_v1",
        "benchmark": "benchmarks/m06_adaptive@v1",
        "hypothesis": "Routing no-retrieval, single-hop, and iterative questions before retrieval should reduce unnecessary calls and improve multi-hop evidence completeness; a qrel-blind retrieval-quality gate should recover stale-source questions through a fallback corpus.",
        "controls": {
            "router_train_split_separate_from_heldout": True,
            "primary_retrieval": "BM25 top-1 per step",
            "max_iterative_steps": 2,
            "fallback_source": "separate controlled corpus",
            "qrels_exposed_to_runtime": False,
            "oracle_route_is_diagnostic_ceiling_only": True,
        },
        "systems": {system["name"]: system for system in systems},
        "warning": "Tiny controlled benchmark. The fallback corpus simulates CRAG-style external correction without making network calls; the oracle route is only a diagnostic ceiling and is not a deployable system.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "control.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# M06 Adaptive/Corrective Control — Phase 1",
        "",
        "| System | Route acc | Evidence recall | Evidence complete | Mean calls | Unnecessary retrieval | Iterative under-route | Correction P | Correction R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in systems:
        metrics = system["metrics"]
        lines.append(
            f"| {system['name']} | {metrics['route_accuracy']:.3f} | {metrics['evidence_recall']:.3f} | {metrics['evidence_complete']:.3f} | {metrics['mean_retrieval_calls']:.2f} | {metrics['unnecessary_retrieval_rate']:.3f} | {metrics['iterative_under_route_rate']:.3f} | {metrics['correction_precision']:.3f} | {metrics['correction_recall']:.3f} |"
        )
    lines.extend([
        "",
        "Evidence metrics exclude no-retrieval questions. Correction labels evaluate whether the qrel-blind retrieval judge triggers fallback only for the deliberately stale primary-source cases.",
        "",
    ])
    (OUT / "control.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
