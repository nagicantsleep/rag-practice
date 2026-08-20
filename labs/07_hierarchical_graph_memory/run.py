from __future__ import annotations
import json
import time
from pathlib import Path
from statistics import fmean

from rag_practice.evaluation.structured import evidence_complete_at_budget, recall_at_budget, reciprocal_rank, summarize_traces
from rag_practice.ir.bm25 import BM25Index
from rag_practice.structured import (
    GlobalGraphRetriever, HippoRAGRetriever, KAGPathRetriever, KnowledgeGraph,
    MemoryEvent, RaptorStyleIndex, StructuredDocument, TemporalMemoryIndex,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks" / "m07_structured"
RESULTS = Path(__file__).resolve().parent / "results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timed_build(factory):
    start = time.perf_counter()
    obj = factory()
    return obj, (time.perf_counter() - start) * 1000.0


def evaluate_retrieval():
    docs = [StructuredDocument.from_mapping(row) for row in load_jsonl(BENCH / "documents.jsonl")]
    queries = load_jsonl(BENCH / "queries.jsonl")
    graph, graph_build_ms = timed_build(lambda: KnowledgeGraph(docs))
    methods = {}
    build_ms = {}
    methods["flat_bm25"], build_ms["flat_bm25"] = timed_build(lambda: BM25Index({d.id: d.text for d in docs}))
    methods["flat_metadata_bm25"], build_ms["flat_metadata_bm25"] = timed_build(lambda: BM25Index({d.id: f"collection {d.collection} {d.text}" for d in docs}))
    methods["raptor_style"], build_ms["raptor_style"] = timed_build(lambda: RaptorStyleIndex(docs))
    methods["kag_path"] = KAGPathRetriever(graph); build_ms["kag_path"] = graph_build_ms
    methods["graph_global"] = GlobalGraphRetriever(graph); build_ms["graph_global"] = graph_build_ms
    methods["hipporag_ppr"] = HippoRAGRetriever(graph); build_ms["hipporag_ppr"] = graph_build_ms

    systems = {}
    for name, method in methods.items():
        traces = []
        for row in queries:
            start = time.perf_counter()
            ranking = [doc_id for doc_id, _ in method.search(row["query"], k=10)]
            query_ms = (time.perf_counter() - start) * 1000.0
            budget = len(row["relevant"])
            traces.append({
                "query_id": row["id"], "task": row["task"], "query": row["query"],
                "relevant": row["relevant"], "ranking": ranking,
                "recall@3": recall_at_budget(ranking, row["relevant"], 3),
                "recall@5": recall_at_budget(ranking, row["relevant"], 5),
                "recall@10": recall_at_budget(ranking, row["relevant"], 10),
                "recall_at_evidence_budget": recall_at_budget(ranking, row["relevant"], budget),
                "evidence_complete_at_budget": evidence_complete_at_budget(ranking, row["relevant"], budget),
                "reciprocal_rank": reciprocal_rank(ranking, row["relevant"]),
                "query_ms": query_ms,
            })
        systems[name] = {"build_ms": build_ms[name], "metrics": summarize_traces(traces), "traces": traces}

    systems["raptor_style"]["structure"] = methods["raptor_style"].stats()
    graph_stats = methods["hipporag_ppr"].stats()
    for name in ("kag_path", "graph_global", "hipporag_ppr"):
        systems[name]["structure"] = graph_stats
    return {"benchmark_queries": len(queries), "documents": len(docs), "systems": systems}


def evaluate_memory():
    base_events = [MemoryEvent.from_mapping(row) for row in load_jsonl(BENCH / "memory.jsonl")]
    updates = [MemoryEvent.from_mapping(row) for row in load_jsonl(BENCH / "memory_updates.jsonl")]
    queries = load_jsonl(BENCH / "memory_queries.jsonl")
    all_events = base_events + updates

    flat, flat_build_ms = timed_build(lambda: BM25Index({e.id: e.text for e in all_events}))
    temporal, temporal_build_ms = timed_build(lambda: TemporalMemoryIndex(base_events))
    update_times = []
    for event in updates:
        start = time.perf_counter(); temporal.add(event); update_times.append((time.perf_counter() - start) * 1000.0)

    def run_system(method):
        traces = []
        for row in queries:
            start = time.perf_counter(); ranking = [doc_id for doc_id, _ in method.search(row["query"], k=4)]; query_ms=(time.perf_counter()-start)*1000.0
            hit = float(bool(ranking) and ranking[0] in set(row["relevant"]))
            traces.append({"query_id":row["id"],"task":row["task"],"query":row["query"],"relevant":row["relevant"],"ranking":ranking,"hit@1":hit,"query_ms":query_ms})
        current = [t for t in traces if t["task"] == "memory_current"]
        previous = [t for t in traces if t["task"] == "memory_previous"]
        return {"metrics": {
            "hit@1": fmean(t["hit@1"] for t in traces),
            "current_hit@1": fmean(t["hit@1"] for t in current) if current else 0.0,
            "previous_hit@1": fmean(t["hit@1"] for t in previous) if previous else 0.0,
            "stale_current_rate": 1.0 - (fmean(t["hit@1"] for t in current) if current else 0.0),
            "mean_query_ms": fmean(t["query_ms"] for t in traces),
        }, "traces": traces}

    flat_result = run_system(flat); flat_result["build_ms"] = flat_build_ms
    temporal_result = run_system(temporal); temporal_result["build_ms"] = temporal_build_ms; temporal_result["mean_update_ms"] = fmean(update_times); temporal_result["structure"] = temporal.stats()
    return {"base_events":len(base_events),"updates":len(updates),"systems":{"flat_bm25_all_versions":flat_result,"temporal_memory":temporal_result}}


def render_markdown(payload: dict) -> str:
    lines = ["# M07 structured retrieval results", "", "## Retrieval / graph / hierarchy", "", "| System | Recall@3 | Recall@5 | Evidence complete@budget | MRR | Mean query ms |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in payload["retrieval"]["systems"].items():
        m=result["metrics"]; lines.append(f"| {name} | {m['mean_recall@3']:.3f} | {m['mean_recall@5']:.3f} | {m['evidence_complete_at_budget']:.3f} | {m['mrr']:.3f} | {m['mean_query_ms']:.3f} |")
    lines += ["", "## Memory / updates", "", "| System | Hit@1 | Current Hit@1 | Previous Hit@1 | Stale current rate | Mean query ms |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in payload["memory"]["systems"].items():
        m=result["metrics"]; lines.append(f"| {name} | {m['hit@1']:.3f} | {m['current_hit@1']:.3f} | {m['previous_hit@1']:.3f} | {m['stale_current_rate']:.3f} | {m['mean_query_ms']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    payload = {
        "hypothesis": "hierarchy helps collection-wide evidence packaging, graph paths help relation chains/global aggregation, and version-aware memory prevents stale current-fact retrieval; no single structure should dominate every query class",
        "controls": {
            "shared_static_corpus": True,
            "flat_bm25_baseline": True,
            "flat_metadata_control": True,
            "gold_triples_isolate_graph_retrieval_from_information_extraction": True,
            "runtime_uses_no_qrels_or_reference_answers": True,
            "generation_not_involved": "M07 isolates structured retrieval/context evidence; answer generation is intentionally not evaluated",
        },
        "retrieval": evaluate_retrieval(),
        "memory": evaluate_memory(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (RESULTS / "results.md").write_text(render_markdown(payload))
    print(render_markdown(payload))

if __name__ == "__main__": main()
