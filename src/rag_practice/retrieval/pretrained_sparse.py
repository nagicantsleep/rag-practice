from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any


def _nonzero_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_nonzero_count(item) for item in value)
    if hasattr(value, "_nnz"):
        try:
            return int(value._nnz())
        except (RuntimeError, TypeError):
            pass
    if hasattr(value, "count_nonzero"):
        count = value.count_nonzero()
        return int(count.item() if hasattr(count, "item") else count)
    raise TypeError(f"cannot count sparse footprint for {type(value)!r}")


def _row_scores(value: Any) -> list[float]:
    row = value[0]
    if hasattr(row, "detach"):
        row = row.detach()
    if hasattr(row, "cpu"):
        row = row.cpu()
    if hasattr(row, "tolist"):
        return [float(score) for score in row.tolist()]
    return [float(score) for score in row]


class SentenceTransformerSparseRetriever:
    """Full pretrained sparse retrieval while keeping ranking explicit.

    Sentence Transformers' ``SparseEncoder`` supplies the learned sparse
    representations and its model-specific similarity function. This wrapper
    owns document IDs, ranking, timing, and evaluation-facing footprint data so
    it can be compared with BM25 and the educational SPLADE-style implementation.
    """

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SparseEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SentenceTransformerSparseRetriever requires the 'pretrained' extra"
            ) from exc

        start = perf_counter()
        self.model = SparseEncoder(model_name, revision=revision, device=device)
        self.model_load_ms = (perf_counter() - start) * 1000
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.document_ids: list[str] = []
        self.document_embeddings: Any | None = None
        self.index_build_ms = 0.0

    def fit(self, documents: Mapping[str, str]) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.document_ids = sorted(documents)
        texts = [documents[document_id] for document_id in self.document_ids]
        start = perf_counter()
        self.document_embeddings = self.model.encode_document(
            texts,
            show_progress_bar=False,
        )
        self.index_build_ms = (perf_counter() - start) * 1000

    def search(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        if self.document_embeddings is None:
            raise RuntimeError("fit must be called before search")
        if k <= 0:
            return []
        query_embedding = self.model.encode_query([query], show_progress_bar=False)
        scores = _row_scores(
            self.model.similarity(query_embedding, self.document_embeddings)
        )
        if len(scores) != len(self.document_ids):
            raise ValueError("similarity output does not match document count")
        ranked = list(zip(self.document_ids, scores, strict=True))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def nonzero_document_values(self) -> int:
        if self.document_embeddings is None:
            raise RuntimeError("fit must be called first")
        return _nonzero_count(self.document_embeddings)

    def mean_nonzero_document_values(self) -> float:
        if not self.document_ids:
            raise RuntimeError("fit must be called first")
        return self.nonzero_document_values() / len(self.document_ids)
