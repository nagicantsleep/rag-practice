from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

from rag_practice.core.chunking import FixedSizeChunker
from rag_practice.core.models import Document
from rag_practice.evaluation.chunking import evaluate_chunk_rankings
from rag_practice.indexing.chunking import (
    MetadataEnrichedChunker,
    ParagraphChunker,
    SemanticChunker,
    SentenceChunker,
)
from rag_practice.ir.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    document_rows = load_jsonl(ROOT / "benchmarks/m03_chunking/documents.jsonl")
    queries = load_jsonl(ROOT / "benchmarks/m03_chunking/queries.jsonl")
    documents = [
        Document(row["id"], row["text"], metadata=row["metadata"])
        for row in document_rows
    ]

    strategies = {
        "fixed_24": FixedSizeChunker(chunk_size_words=24, overlap_words=0),
        "fixed_24_overlap_8": FixedSizeChunker(chunk_size_words=24, overlap_words=8),
        "sentence_35": SentenceChunker(max_words=35),
        "paragraph_80": ParagraphChunker(max_words=80),
        "semantic_50": SemanticChunker(max_words=50, similarity_threshold=0.08),
        "sentence_35_metadata": MetadataEnrichedChunker(
            SentenceChunker(max_words=35),
            fields=("title", "section", "tags", "region"),
        ),
    }

    results = []
    for name, chunker in strategies.items():
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

        metrics = evaluate_chunk_rankings(rankings, by_id, queries, k=3)
        results.append(
            {
                "strategy": name,
                "chunks": len(chunks),
                "mean_chunk_words": fmean(len(chunk.text.split()) for chunk in chunks),
                "build_ms": build_ms,
                "mean_query_ms": fmean(query_times),
                "metrics": metrics,
                "top3": rankings,
            }
        )

    output = {
        "experiment_id": "m03_chunking_phase1_v1",
        "retriever": "BM25 with fixed parameters over chunk text",
        "benchmark": "benchmarks/m03_chunking@v1",
        "hypothesis": "Boundary-aware chunking should improve evidence completeness while overlap trades extra coverage for duplicated context; metadata should help metadata-dependent queries.",
        "results": results,
    }
    output_dir = ROOT / "labs/03_indexing_chunking/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1.json").write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "# M03 Chunking Phase 1",
        "",
        "Retriever is fixed to BM25; only the chunk/index representation changes.",
        "",
        "| Strategy | Chunks | Mean words | Doc hit@1 | Evidence@1 | Evidence@3 | Source token util@3 | Relevant context@3 | Mean query ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        m = item["metrics"]
        lines.append(
            f"| {item['strategy']} | {item['chunks']} | {item['mean_chunk_words']:.1f} | "
            f"{m['document_hit@1']:.3f} | {m['evidence_complete@1']:.3f} | "
            f"{m['evidence_complete@3']:.3f} | {m['source_token_utilization@3']:.3f} | "
            f"{m['relevant_context_fraction@3']:.3f} | {item['mean_query_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`source_token_utilization@3` is the number of unique source-word positions represented in the retrieved context divided by the actual context word count. It therefore penalizes overlap and metadata prefix overhead instead of treating duplicated tokens as free.",
            "",
        ]
    )
    (output_dir / "phase1.md").write_text("\n".join(lines))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
