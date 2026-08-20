from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

from rag_practice.core.chunking import FixedSizeChunker
from rag_practice.core.models import Document
from rag_practice.embeddings.hashing import HashingEmbedder
from rag_practice.evaluation.rag import (
    answer_contains_reference,
    citation_precision,
    citation_recall,
    grounded_token_recall,
    token_f1,
)
from rag_practice.evaluation.retrieval import evaluate_rankings
from rag_practice.generation.extractive import TopChunkExtractiveGenerator
from rag_practice.ir.bm25 import BM25Index
from rag_practice.rag.pipeline import NaiveRAGPipeline

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    corpus_rows = load_jsonl(ROOT / "benchmarks/m00_ir/corpus.jsonl")
    qa_rows = load_jsonl(ROOT / "benchmarks/m01_rag/questions.jsonl")
    documents = [Document(row["id"], row["text"]) for row in corpus_rows]
    corpus = {row["id"]: row["text"] for row in corpus_rows}

    pipeline = NaiveRAGPipeline(
        embedder=HashingEmbedder(256),
        generator=TopChunkExtractiveGenerator(),
        chunker=FixedSizeChunker(80, 0),
    )
    pipeline.index_documents(documents)
    bm25 = BM25Index(corpus)

    rankings: dict[str, list[str]] = {}
    bm25_rankings: dict[str, list[str]] = {}
    per_query: list[dict] = []
    no_retrieval_contains: list[float] = []

    for row in qa_rows:
        trace = pipeline.answer(row["question"], top_k=3)
        no_context_answer = pipeline.generator.generate(
            question=row["question"], prompt="No context provided.", retrieved=[]
        )
        no_retrieval_contains.append(
            answer_contains_reference(no_context_answer.text, row["reference"])
        )

        ranked_docs = [item.chunk.document_id for item in trace.retrieved]
        rankings[row["id"]] = ranked_docs
        bm25_rankings[row["id"]] = [
            document_id for document_id, _ in bm25.search(row["question"], k=3)
        ]

        cited_docs = [chunk_id.split("::", 1)[0] for chunk_id in trace.answer.cited_chunk_ids]
        cited_texts = [
            item.chunk.text
            for item in trace.retrieved
            if item.chunk.id in trace.answer.cited_chunk_ids
        ]
        relevant = set(row["relevant_document_ids"])
        per_query.append(
            {
                "id": row["id"],
                "question": row["question"],
                "reference": row["reference"],
                "retrieved_document_ids": ranked_docs,
                "top_document_id": ranked_docs[0] if ranked_docs else None,
                "answer": trace.answer.text,
                "cited_document_ids": cited_docs,
                "answer_token_f1": token_f1(trace.answer.text, row["reference"]),
                "answer_contains_reference": answer_contains_reference(
                    trace.answer.text, row["reference"]
                ),
                "grounded_token_recall": grounded_token_recall(
                    trace.answer.text, cited_texts
                ),
                "citation_precision": citation_precision(cited_docs, relevant),
                "citation_recall": citation_recall(cited_docs, relevant),
                "timings_ms": trace.timings_ms,
                "prompt_tokens": trace.prompt_tokens,
                "output_tokens": trace.output_tokens,
            }
        )

    qrels = {
        row["id"]: {document_id: 1.0 for document_id in row["relevant_document_ids"]}
        for row in qa_rows
    }
    result = {
        "experiment_id": "m01_naive_rag_v1",
        "benchmark": "benchmarks/m01_rag@v1",
        "method": {
            "chunk_size_words": 80,
            "overlap_words": 0,
            "embedder": "HashingEmbedder",
            "dimensions": 256,
            "top_k": 3,
            "generator": "TopChunkExtractiveGenerator",
        },
        "retrieval": {
            "hashing_vector": evaluate_rankings(rankings, qrels, ks=(1, 3)),
            "bm25_baseline": evaluate_rankings(bm25_rankings, qrels, ks=(1, 3)),
        },
        "generation": {
            "mean_answer_token_f1": fmean(item["answer_token_f1"] for item in per_query),
            "answer_contains_reference_rate": fmean(
                item["answer_contains_reference"] for item in per_query
            ),
            "mean_grounded_token_recall": fmean(
                item["grounded_token_recall"] for item in per_query
            ),
            "mean_citation_precision": fmean(
                item["citation_precision"] for item in per_query
            ),
            "mean_citation_recall": fmean(item["citation_recall"] for item in per_query),
            "no_retrieval_answer_contains_reference_rate": fmean(no_retrieval_contains),
        },
        "system": {
            "mean_end_to_end_ms": fmean(
                item["timings_ms"]["end_to_end"] for item in per_query
            ),
            "mean_retrieval_ms": fmean(
                item["timings_ms"]["retrieval"] for item in per_query
            ),
            "mean_prompt_tokens": fmean(item["prompt_tokens"] for item in per_query),
            "mean_output_tokens": fmean(item["output_tokens"] for item in per_query),
        },
        "per_query": per_query,
    }

    output = ROOT / "labs/01_naive_rag/results/baseline.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
