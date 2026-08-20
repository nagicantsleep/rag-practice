from __future__ import annotations

from collections.abc import Iterable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("LateInteractionRetriever requires PyTorch") from exc

from rag_practice.ir.text import tokenize
from rag_practice.retrieval.neural_dual_encoder import TrainingPair


class _TokenEncoder(nn.Module):
    def __init__(self, vocab_size: int, dimensions: int) -> None:
        super().__init__()
        self.query_embeddings = nn.Embedding(vocab_size, dimensions)
        self.document_embeddings = nn.Embedding(vocab_size, dimensions)

    def query(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_embeddings(ids), dim=-1)

    def document(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.document_embeddings(ids), dim=-1)


class LateInteractionRetriever:
    """Educational ColBERT-style token-level MaxSim retriever.

    Query and document tokens are independently encoded. Relevance is the sum,
    over query tokens, of each token's maximum similarity to any document token.
    A tiny learned embedding table intentionally replaces ColBERT's contextual
    BERT encoder so the late-interaction operation remains easy to inspect.
    """

    def __init__(self, dimensions: int = 16, seed: int = 7) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.seed = seed
        self.vocabulary: dict[str, int] = {"<unk>": 0}
        self.document_ids: list[str] = []
        self._document_ids_tensor: torch.Tensor | None = None
        self._document_mask: torch.Tensor | None = None
        self._model: _TokenEncoder | None = None

    def _build_vocabulary(self, texts: Iterable[str]) -> None:
        terms = sorted({term for text in texts for term in tokenize(text)})
        self.vocabulary = {"<unk>": 0}
        for term in terms:
            if term not in self.vocabulary:
                self.vocabulary[term] = len(self.vocabulary)

    def _token_ids(self, text: str) -> list[int]:
        ids = [self.vocabulary.get(term, 0) for term in tokenize(text)]
        return ids or [0]

    def _pad(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self._token_ids(text) for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for row_index, row in enumerate(rows):
            ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[row_index, : len(row)] = True
        return ids, mask

    def _score_batch(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        document_ids: torch.Tensor,
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self._model is not None
        query_vectors = self._model.query(query_ids)
        document_vectors = self._model.document(document_ids)
        similarities = torch.einsum(
            "bqd,nld->bnql", query_vectors, document_vectors
        )
        similarities = similarities.masked_fill(
            ~document_mask[None, :, None, :], float("-inf")
        )
        maxsim = similarities.max(dim=-1).values
        maxsim = maxsim.masked_fill(~query_mask[:, None, :], 0.0)
        return maxsim.sum(dim=-1)

    def fit(
        self,
        documents: dict[str, str],
        pairs: list[TrainingPair],
        *,
        epochs: int = 100,
        learning_rate: float = 0.05,
        temperature: float = 0.15,
    ) -> list[float]:
        if not documents or not pairs:
            raise ValueError("documents and pairs must not be empty")
        if epochs <= 0 or learning_rate <= 0 or temperature <= 0:
            raise ValueError("epochs, learning_rate, and temperature must be positive")

        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.document_ids = sorted(documents)
        document_texts = [documents[document_id] for document_id in self.document_ids]
        query_texts = [pair.query for pair in pairs]
        self._build_vocabulary([*document_texts, *query_texts])
        self._model = _TokenEncoder(len(self.vocabulary), self.dimensions)
        self._document_ids_tensor, self._document_mask = self._pad(document_texts)
        query_ids, query_mask = self._pad(query_texts)

        positions = {document_id: index for index, document_id in enumerate(self.document_ids)}
        labels = torch.tensor(
            [positions[pair.document_id] for pair in pairs], dtype=torch.long
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        losses: list[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            scores = self._score_batch(
                query_ids,
                query_mask,
                self._document_ids_tensor,
                self._document_mask,
            ) / temperature
            loss = F.cross_entropy(scores, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return losses

    def search(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        if (
            self._model is None
            or self._document_ids_tensor is None
            or self._document_mask is None
        ):
            raise RuntimeError("fit must be called before search")
        if k <= 0:
            return []
        query_ids, query_mask = self._pad([query])
        with torch.no_grad():
            scores = self._score_batch(
                query_ids,
                query_mask,
                self._document_ids_tensor,
                self._document_mask,
            )[0]
        ranked = [
            (document_id, float(scores[index]))
            for index, document_id in enumerate(self.document_ids)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]
