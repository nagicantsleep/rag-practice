from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from rag_practice.retrieval.pretrained import SentenceTransformerRetriever


class _FakeSentenceTransformer:
    def __init__(self, model_name, revision=None, device="cpu") -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(self, *, inputs, **kwargs):
        rows = []
        for text in inputs:
            if "alpha" in text:
                rows.append([1.0, 0.0])
            elif "beta" in text:
                rows.append([0.0, 1.0])
            else:
                rows.append([0.5, 0.5])
        values = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)


def test_pretrained_wrapper_ranks_with_cosine_dot_product(monkeypatch) -> None:
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    retriever = SentenceTransformerRetriever(
        "fake/model", revision="abc123", device="cpu"
    )
    retriever.fit({"d2": "beta evidence", "d1": "alpha evidence"})

    assert retriever.dimensions == 2
    assert retriever.search("alpha question", k=2)[0][0] == "d1"
    assert retriever.logical_index_bytes() == 2 * 2 * 4


def test_pretrained_wrapper_validates_index_state(monkeypatch) -> None:
    fake_module = ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    retriever = SentenceTransformerRetriever("fake/model")
    try:
        retriever.search("query")
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("search before fit must fail")
