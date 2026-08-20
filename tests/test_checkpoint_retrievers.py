from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from rag_practice.retrieval.pretrained_colbert import PyLateColBERTRetriever
from rag_practice.retrieval.pretrained_sparse import SentenceTransformerSparseRetriever


class _FakeSparseEncoder:
    def __init__(self, model_name, revision=None, device="cpu") -> None:
        pass

    def encode_document(self, texts, **kwargs):
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "alpha" in text else [0.0, 1.0])
        return torch.tensor(rows)

    def encode_query(self, texts, **kwargs):
        rows = []
        for text in texts:
            rows.append([1.0, 0.0] if "alpha" in text else [0.0, 1.0])
        return torch.tensor(rows)

    def similarity(self, queries, documents):
        return queries @ documents.T


class _FakeColBERT:
    def __init__(self, model_name_or_path) -> None:
        pass

    def encode(self, texts, *, is_query, **kwargs):
        if is_query:
            return [torch.ones((2, 4)) for _ in texts]
        return [torch.ones((3, 4)) for _ in texts]


class _FakeRank:
    @staticmethod
    def rerank(*, documents_ids, queries_embeddings, documents_embeddings):
        ids = documents_ids[0]
        return [[{"id": ids[-1], "score": 2.0}, {"id": ids[0], "score": 1.0}]]


def test_sparse_checkpoint_wrapper_ranks_explicitly(monkeypatch) -> None:
    fake = ModuleType("sentence_transformers")
    fake.SparseEncoder = _FakeSparseEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    retriever = SentenceTransformerSparseRetriever("fake/splade")
    retriever.fit({"d2": "beta document", "d1": "alpha document"})

    assert retriever.search("alpha query", k=2)[0][0] == "d1"
    assert retriever.nonzero_document_values() == 2
    assert retriever.mean_nonzero_document_values() == 1.0


def test_colbert_checkpoint_wrapper_uses_pylate_reranking(monkeypatch) -> None:
    fake = ModuleType("pylate")
    fake.models = SimpleNamespace(ColBERT=_FakeColBERT)
    fake.rank = _FakeRank
    monkeypatch.setitem(sys.modules, "pylate", fake)

    retriever = PyLateColBERTRetriever("fake/colbert")
    retriever.fit({"d2": "second", "d1": "first"})
    results = retriever.search("query", k=1)

    assert results == [("d2", 2.0)]
    assert retriever.document_token_vectors() == 6
    assert retriever.logical_embedding_bytes() == 2 * 3 * 4 * 4
