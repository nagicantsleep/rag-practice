from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag_practice.core.models import Chunk, Document
from rag_practice.indexing.chunking import ParagraphChunker, SentenceChunker
from rag_practice.ir.bm25 import BM25Index


def _metadata_text(document: Document) -> str:
    parts: list[str] = []
    for field in ("title", "section", "tags", "region"):
        value = document.metadata.get(field)
        if value in (None, "", []):
            continue
        if isinstance(value, (list, tuple, set)):
            rendered = " ".join(str(item) for item in value)
        else:
            rendered = str(value)
        parts.append(f"{field} {rendered}")
    return " ".join(parts)


def _span_overlap(left: Chunk, right: Chunk) -> int:
    if left.document_id != right.document_id:
        return 0
    return max(0, min(left.end_word, right.end_word) - max(left.start_word, right.start_word))


class ParentChildBM25Index:
    """Retrieve narrow sentence children and return their wider paragraph parents."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        parent_chunker: ParagraphChunker | None = None,
        child_chunker: SentenceChunker | None = None,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.parent_chunker = parent_chunker or ParagraphChunker(max_words=40)
        self.child_chunker = child_chunker or SentenceChunker(max_words=18)
        self.parents = self.parent_chunker.chunk_many(list(documents))
        self.children = self.child_chunker.chunk_many(list(documents))
        self.parent_by_id = {chunk.id: chunk for chunk in self.parents}
        self.child_by_id = {chunk.id: chunk for chunk in self.children}

        parents_by_document: defaultdict[str, list[Chunk]] = defaultdict(list)
        for parent in self.parents:
            parents_by_document[parent.document_id].append(parent)

        self.child_to_parent: dict[str, str] = {}
        for child in self.children:
            candidates = parents_by_document[child.document_id]
            parent = max(candidates, key=lambda item: (_span_overlap(child, item), -item.start_word))
            if _span_overlap(child, parent) <= 0:
                raise ValueError(f"no parent span found for {child.id}")
            self.child_to_parent[child.id] = parent.id

        self.child_index = BM25Index({chunk.id: chunk.text for chunk in self.children})

    def search(self, query: str, *, k: int = 3, child_k: int = 8) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        child_results = self.child_index.search(query, k=max(k, child_k))
        parent_scores: defaultdict[str, float] = defaultdict(float)
        for child_id, score in child_results:
            parent_scores[self.child_to_parent[child_id]] += score
        ranked = sorted(parent_scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def searchable_index_words(self) -> int:
        return sum(len(chunk.text.split()) for chunk in self.children)

    def stored_context_words(self) -> int:
        return sum(len(chunk.text.split()) for chunk in self.parents)


class HierarchicalBM25Index:
    """Route at document level using metadata+body, then rank plain sentence leaves."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        leaf_chunker: SentenceChunker | None = None,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.documents = list(documents)
        self.leaf_chunker = leaf_chunker or SentenceChunker(max_words=35)
        self.leaves = self.leaf_chunker.chunk_many(self.documents)
        self.leaf_by_id = {chunk.id: chunk for chunk in self.leaves}
        self.root_texts = {
            document.id: f"{_metadata_text(document)} {document.text}".strip()
            for document in self.documents
        }
        self.root_index = BM25Index(self.root_texts)
        self.leaf_index = BM25Index({chunk.id: chunk.text for chunk in self.leaves})

    def route(self, query: str, *, k: int = 2) -> list[tuple[str, float]]:
        return self.root_index.search(query, k=k)

    def search(self, query: str, *, k: int = 3, route_k: int = 2) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        routes = self.route(query, k=route_k)
        if not routes:
            return []
        route_scores = dict(routes)
        allowed = set(route_scores)
        leaf_results = self.leaf_index.search(query, k=len(self.leaves))
        combined = [
            (chunk_id, score + route_scores[self.leaf_by_id[chunk_id].document_id])
            for chunk_id, score in leaf_results
            if self.leaf_by_id[chunk_id].document_id in allowed
        ]
        combined.sort(key=lambda item: (-item[1], item[0]))
        return combined[:k]

    def searchable_index_words(self) -> int:
        return sum(len(text.split()) for text in self.root_texts.values()) + sum(
            len(chunk.text.split()) for chunk in self.leaves
        )
