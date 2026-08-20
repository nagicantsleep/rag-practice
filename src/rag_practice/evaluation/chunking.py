from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_practice.core.models import Chunk


def source_token_utilization(chunks: Sequence[Chunk]) -> float:
    total_context_words = sum(len(chunk.text.split()) for chunk in chunks)
    if total_context_words == 0:
        return 0.0
    positions: set[tuple[str, int]] = set()
    for chunk in chunks:
        positions.update(
            (chunk.document_id, position)
            for position in range(chunk.start_word, chunk.end_word)
        )
    return len(positions) / total_context_words


def evaluate_chunk_rankings(
    rankings: Mapping[str, list[str]],
    chunks: Mapping[str, Chunk],
    queries: Sequence[dict],
    *,
    k: int = 3,
) -> dict[str, float]:
    if not queries:
        raise ValueError("queries must not be empty")

    doc_hit_1 = 0
    doc_hit_k = 0
    evidence_1 = 0
    evidence_k = 0
    utilization = 0.0
    relevant_fraction = 0.0

    for query in queries:
        ranked_ids = rankings.get(query["id"], [])
        relevant_document = query["relevant_document_id"]
        required = [phrase.lower() for phrase in query.get("required_phrases", [])]

        top1 = [chunks[item] for item in ranked_ids[:1] if item in chunks]
        topk = [chunks[item] for item in ranked_ids[:k] if item in chunks]
        if top1 and top1[0].document_id == relevant_document:
            doc_hit_1 += 1
        if any(chunk.document_id == relevant_document for chunk in topk):
            doc_hit_k += 1

        def complete(items: Sequence[Chunk]) -> bool:
            relevant_text = " ".join(
                chunk.text.lower()
                for chunk in items
                if chunk.document_id == relevant_document
            )
            return bool(required) and all(phrase in relevant_text for phrase in required)

        evidence_1 += int(complete(top1))
        evidence_k += int(complete(topk))
        utilization += source_token_utilization(topk)
        total_words = sum(len(chunk.text.split()) for chunk in topk)
        relevant_words = sum(
            len(chunk.text.split())
            for chunk in topk
            if chunk.document_id == relevant_document
        )
        relevant_fraction += relevant_words / total_words if total_words else 0.0

    count = len(queries)
    return {
        "document_hit@1": doc_hit_1 / count,
        f"document_hit@{k}": doc_hit_k / count,
        "evidence_complete@1": evidence_1 / count,
        f"evidence_complete@{k}": evidence_k / count,
        f"source_token_utilization@{k}": utilization / count,
        f"relevant_context_fraction@{k}": relevant_fraction / count,
    }
