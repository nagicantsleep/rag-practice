from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np


class SentenceTransformerRetriever:
    """Thin, inspectable cosine retriever around a pretrained SentenceTransformer.

    The wrapper deliberately owns ranking and document storage instead of using a
    vector-database abstraction. This keeps the retrieval mechanics comparable
    with the earlier M00-M02 implementations while allowing a broadly pretrained
    semantic encoder to provide the representations.
    """

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SentenceTransformerRetriever requires the 'pretrained' extra"
            ) from exc

        start = perf_counter()
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
        )
        self.model_load_ms = (perf_counter() - start) * 1000
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.document_ids: list[str] = []
        self.document_vectors: np.ndarray | None = None
        self.index_build_ms = 0.0

    @property
    def dimensions(self) -> int:
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("model did not report an embedding dimension")
        return int(dimension)

    def fit(self, documents: Mapping[str, str]) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        self.document_ids = sorted(documents)
        texts = [documents[document_id] for document_id in self.document_ids]
        start = perf_counter()
        vectors = self.model.encode(
            inputs=texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.index_build_ms = (perf_counter() - start) * 1000
        self.document_vectors = np.asarray(vectors, dtype=np.float32)
        if self.document_vectors.ndim != 2:
            raise ValueError("model.encode must return a 2D document matrix")
        if self.document_vectors.shape[0] != len(self.document_ids):
            raise ValueError("embedding count does not match document count")

    def search(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        if self.document_vectors is None:
            raise RuntimeError("fit must be called before search")
        if k <= 0:
            return []
        vector = self.model.encode(
            inputs=[query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(vector, dtype=np.float32)[0]
        scores = self.document_vectors @ query_vector
        ranked = [
            (document_id, float(scores[index]))
            for index, document_id in enumerate(self.document_ids)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def logical_index_bytes(self) -> int:
        if self.document_vectors is None:
            raise RuntimeError("fit must be called first")
        return int(self.document_vectors.nbytes)
