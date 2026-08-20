from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.core.chunking import FixedSizeChunker
from rag_practice.core.models import Chunk, Document
from rag_practice.evaluation.chunking import evaluate_chunk_rankings
from rag_practice.indexing.chunking import MetadataEnrichedChunker, SentenceChunker
from rag_practice.indexing.hierarchy import HierarchicalBM25Index, ParentChildBM25Index
from rag_practice.ir.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate_flat(name: str, chunker, documents: list[Document], queries: list[dict]) -> dict:
    start = perf_counter()
    chunks = chunker.chunk_many(documents)
    index = BM25Index({chunk.id: chunk.text for chunk in chunks})
    build_ms = (perf_counter() - start) * 1000
    by_id = {chunk.id: chunk for chunk in chunks}
    rankings: dict[str, list[str]] = {}
    query_times = []
    for query in queries:
        start = perf_counter()
        ranked = index.search(query["query"], k=3)
        query_times.append((perf_counter() - start) * 1000)
        rankings[query["id"]] = [chunk_id for chunk_id, _ in ranked]
    return {
        "strategy": name,
        "returned_representation": "chunk",
        "build_ms": build_ms,
        "mean_query_ms": fmean(query_times),
        "searchable_index_words": sum(len(chunk.text.split()) for chunk in chunks),
        "metrics": evaluate_chunk_rankings(rankings, by_id, queries, k=3),
        "top3": rankings,
    }


def evaluate_parent_child(documents: list[Document], queries: list[dict]) -> dict:
    start = perf_counter()
    index = ParentChildBM25Index(documents)
    build_ms = (perf_counter() - start) * 1000
    rankings: dict[str, list[str]] = {}
    query_times = []
    for query in queries:
        start = perf_counter()
        ranked = index.search(query["query"], k=3)
        query_times.append((perf_counter() - start) * 1000)
        rankings[query["id"]] = [chunk_id for chunk_id, _ in ranked]
    return {
        "strategy": "parent_child",
        "returned_representation": "paragraph parent",
        "build_ms": build_ms,
        "mean_query_ms": fmean(query_times),
        "searchable_index_words": index.searchable_index_words(),
        "stored_context_words": index.stored_context_words(),
        "metrics": evaluate_chunk_rankings(rankings, index.parent_by_id, queries, k=3),
        "top3": rankings,
    }


def evaluate_hierarchical(documents: list[Document], queries: list[dict]) -> dict:
    start = perf_counter()
    index = HierarchicalBM25Index(documents)
    build_ms = (perf_counter() - start) * 1000
    rankings: dict[str, list[str]] = {}
    query_times = []
    route_hits = 0
    routes: dict[str, list[str]] = {}
    for query in queries:
        route = index.route(query["query"], k=2)
        routes[query["id"]] = [document_id for document_id, _ in route]
        if route and route[0][0] == query["relevant_document_id"]:
            route_hits += 1
        start = perf_counter()
        ranked = index.search(query["query"], k=3, route_k=2)
        query_times.append((perf_counter() - start) * 1000)
        rankings[query["id"]] = [chunk_id for chunk_id, _ in ranked]
    metrics = evaluate_chunk_rankings(rankings, index.leaf_by_id, queries, k=3)
    metrics["route_hit@1"] = route_hits / len(queries)
    return {
        "strategy": "hierarchical_metadata_root",
        "returned_representation": "plain sentence leaf",
        "build_ms": build_ms,
        "mean_query_ms": fmean(query_times),
        "searchable_index_words": index.searchable_index_words(),
        "metrics": metrics,
        "routes": routes,
        "top3": rankings,
    }


def main() -> None:
    document_rows = load_jsonl(ROOT / "benchmarks/m03_chunking/documents.jsonl")
    queries = load_jsonl(ROOT / "benchmarks/m03_chunking/queries.jsonl")
    documents = [
        Document(row["id"], row["text"], metadata=row["metadata"])
        for row in document_rows
    ]

    results = [
        evaluate_flat(
            "fixed_24_overlap_8",
            FixedSizeChunker(chunk_size_words=24, overlap_words=8),
            documents,
            queries,
        ),
        evaluate_flat(
            "sentence_35_metadata",
            MetadataEnrichedChunker(
                SentenceChunker(max_words=35),
                fields=("title", "section", "tags", "region"),
            ),
            documents,
            queries,
        ),
        evaluate_parent_child(documents, queries),
        evaluate_hierarchical(documents, queries),
    ]

    output = {
        "experiment_id": "m03_hierarchy_phase2_v1",
        "benchmark": "benchmarks/m03_chunking@v1",
        "retriever": "BM25 at every searchable layer",
        "warning": "Tiny controlled benchmark: use the result to understand representation trade-offs, not as a universal ranking of chunking methods.",
        "hypothesis": "Parent expansion should improve single-hit evidence completeness, while metadata-at-root hierarchical routing should recover metadata-dependent document selection without repeating metadata in returned context chunks.",
        "results": results,
    }

    output_dir = ROOT / "labs/03_indexing_chunking/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase2.json").write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "# M03 Parent-Child and Hierarchical Retrieval",
        "",
        "BM25 scoring is retained at every searchable layer so this phase isolates representation/routing choices.",
        "",
        "| Strategy | Returned context | Doc hit@1 | Evidence@1 | Evidence@3 | Source token util@3 | Relevant context@3 | Searchable index words | Route hit@1 | Mean query ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        m = item["metrics"]
        route = m.get("route_hit@1")
        route_text = f"{route:.3f}" if route is not None else "—"
        lines.append(
            f"| {item['strategy']} | {item['returned_representation']} | {m['document_hit@1']:.3f} | "
            f"{m['evidence_complete@1']:.3f} | {m['evidence_complete@3']:.3f} | "
            f"{m['source_token_utilization@3']:.3f} | {m['relevant_context_fraction@3']:.3f} | "
            f"{item['searchable_index_words']} | {route_text} | {item['mean_query_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Parent-child indexes narrow children but returns wider parents. Hierarchical routing stores a document-level metadata+body root representation and returns plain sentence leaves, so metadata can affect routing without consuming answer-context tokens.",
            "",
        ]
    )
    (output_dir / "phase2.md").write_text("\n".join(lines))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
