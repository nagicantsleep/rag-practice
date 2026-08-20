from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any


def _logical_bytes(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_logical_bytes(item) for item in value)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    raise TypeError(f"cannot calculate embedding footprint for {type(value)!r}")


def _token_vectors(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_token_vectors(item) for item in value)
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        raise TypeError(f"cannot calculate token-vector count for {type(value)!r}")
    return int(shape[-2])


class PyLateColBERTRetriever:
    """Exhaustive ColBERT late-interaction scorer for a small candidate corpus.

    PyLate supplies a pretrained ColBERT encoder and MaxSim reranker. For this
    educational benchmark every document is a candidate, so no PLAID/ANN index
    is involved. That keeps checkpoint quality separate from production index
    approximations and makes the multi-vector representation cost explicit.
    """

    def __init__(self, model_name_or_path: str) -> None:
        try:
            from pylate import models, rank
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyLateColBERTRetriever requires the 'colbert' extra") from exc

        start = perf_counter()
        self.model = models.ColBERT(model_name_or_path=model_name_or_path)
        self.model_load_ms = (perf_counter() - start) * 1000
        self.rank = rank
        self.model_name_or_path = model_name_or_path
        self.document_ids: list[str] = []
        self.document_embeddings: Any | None = None
        self.index_build_ms = 0.0

    def fit(self, documents: Mapping[str, str]) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.document_ids = sorted(documents)
        texts = [documents[document_id] for document_id in self.document_ids]
        start = perf_counter()
        self.document_embeddings = self.model.encode(
            texts,
            is_query=False,
            show_progress_bar=False,
        )
        self.index_build_ms = (perf_counter() - start) * 1000

    def search(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        if self.document_embeddings is None:
            raise RuntimeError("fit must be called before search")
        if k <= 0:
            return []
        query_embeddings = self.model.encode(
            [query],
            is_query=True,
            show_progress_bar=False,
        )
        reranked = self.rank.rerank(
            documents_ids=[self.document_ids],
            queries_embeddings=query_embeddings,
            documents_embeddings=[self.document_embeddings],
        )
        if not reranked:
            return []
        results = [
            (str(item["id"]), float(item["score"])) for item in reranked[0]
        ]
        return results[:k]

    def logical_embedding_bytes(self) -> int:
        if self.document_embeddings is None:
            raise RuntimeError("fit must be called first")
        return _logical_bytes(self.document_embeddings)

    def document_token_vectors(self) -> int:
        if self.document_embeddings is None:
            raise RuntimeError("fit must be called first")
        return _token_vectors(self.document_embeddings)
