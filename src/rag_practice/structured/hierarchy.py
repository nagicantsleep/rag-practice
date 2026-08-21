from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from rag_practice.ir.bm25 import BM25Index
from rag_practice.ir.text import tokenize
from .models import StructuredDocument

@dataclass(frozen=True)
class HierarchyNode:
    id: str
    collection: str
    level: int
    text: str
    document_ids: tuple[str, ...]

class RaptorStyleIndex:
    """Inspectable extractive hierarchy: leaves -> deterministic groups -> collection roots.

    This keeps the RAPTOR mechanism (recursive summaries + hierarchical routing) visible while
    deliberately replacing learned clustering/generative summaries with deterministic source
    collection grouping and extractive summaries.
    """
    def __init__(self, documents: list[StructuredDocument], *, branching_factor: int = 3):
        if not documents:
            raise ValueError("documents must not be empty")
        if branching_factor < 2:
            raise ValueError("branching_factor must be >= 2")
        self.documents = list(documents)
        self.by_id = {d.id: d for d in documents}
        self.branching_factor = branching_factor
        grouped: dict[str, list[StructuredDocument]] = defaultdict(list)
        for doc in documents:
            grouped[doc.collection].append(doc)
        self.groups: dict[str, list[HierarchyNode]] = {}
        self.roots: dict[str, HierarchyNode] = {}
        for collection, docs in grouped.items():
            docs = sorted(docs, key=lambda d: d.id)
            nodes: list[HierarchyNode] = []
            for i in range(0, len(docs), branching_factor):
                batch = docs[i:i+branching_factor]
                nodes.append(HierarchyNode(
                    id=f"{collection}:group:{i//branching_factor}", collection=collection, level=1,
                    text=" ".join(d.text for d in batch), document_ids=tuple(d.id for d in batch),
                ))
            self.groups[collection] = nodes
            self.roots[collection] = HierarchyNode(
                id=f"{collection}:root", collection=collection, level=2,
                text=f"collection {collection} " + " ".join(node.text for node in nodes),
                document_ids=tuple(d.id for d in docs),
            )
        self.root_index = BM25Index({name: node.text for name, node in self.roots.items()})
        self.group_indexes = {
            collection: BM25Index({node.id: node.text for node in nodes})
            for collection, nodes in self.groups.items()
        }
        self.group_by_id = {node.id: node for nodes in self.groups.values() for node in nodes}

    def _content_query(self, query: str, collection: str) -> str:
        blocked = set(tokenize(collection)) | {"network", "collection", "across"}
        terms = [term for term in tokenize(query) if term not in blocked]
        return " ".join(terms) or query

    def route(self, query: str, *, k: int = 2) -> list[tuple[str, float]]:
        return self.root_index.search(query, k=k)

    def search(self, query: str, *, k: int = 5, route_k: int = 2, group_k: int = 3) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        routes = self.route(query, k=route_k)
        if not routes:
            return []
        scores: dict[str, float] = {}
        for collection, route_score in routes:
            content_query = self._content_query(query, collection)
            group_results = self.group_indexes[collection].search(content_query, k=group_k)
            candidate_ids: set[str] = set()
            group_score_by_doc: dict[str, float] = {}
            for group_id, group_score in group_results:
                node = self.group_by_id[group_id]
                for doc_id in node.document_ids:
                    candidate_ids.add(doc_id)
                    group_score_by_doc[doc_id] = max(group_score_by_doc.get(doc_id, 0.0), group_score)
            if not candidate_ids:
                continue
            leaf_index = BM25Index({doc_id: self.by_id[doc_id].text for doc_id in candidate_ids})
            leaf_scores = dict(leaf_index.search(content_query, k=len(candidate_ids)))
            for doc_id in candidate_ids:
                score = route_score * 0.05 + group_score_by_doc.get(doc_id, 0.0) * 0.15 + leaf_scores.get(doc_id, 0.0)
                scores[doc_id] = max(scores.get(doc_id, 0.0), score)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def stats(self) -> dict[str, int]:
        return {
            "leaf_documents": len(self.documents),
            "summary_nodes": len(self.group_by_id) + len(self.roots),
            "summary_words": sum(len(node.text.split()) for node in self.group_by_id.values()) + sum(len(node.text.split()) for node in self.roots.values()),
        }
