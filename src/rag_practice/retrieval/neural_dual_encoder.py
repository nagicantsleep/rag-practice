from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - environment-dependent guard
    raise ImportError(
        "TinyNeuralDualEncoder requires PyTorch. Install the 'neural' extra."
    ) from exc

from rag_practice.ir.text import tokenize


@dataclass(frozen=True)
class TrainingPair:
    query: str
    document_id: str


class _DualProjection(nn.Module):
    def __init__(self, vocabulary_size: int, dimensions: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(vocabulary_size, dimensions, bias=False)
        self.document_projection = nn.Linear(vocabulary_size, dimensions, bias=False)

    def encode_queries(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.tanh(self.query_projection(features)), dim=-1)

    def encode_documents(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(torch.tanh(self.document_projection(features)), dim=-1)


class TinyNeuralDualEncoder:
    """Small supervised dual encoder used to expose learned dense retrieval.

    Text is represented as a bag-of-words input and projected through separate
    learned query/document encoders. Training uses in-batch/full-corpus softmax
    contrastive loss. This is intentionally tiny and domain-specific; it teaches
    the mechanics of learned dense retrieval without downloading a pretrained
    model.
    """

    def __init__(self, dimensions: int = 32, seed: int = 7) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.seed = seed
        self.vocabulary: dict[str, int] = {"<unk>": 0}
        self.document_ids: list[str] = []
        self._model: _DualProjection | None = None
        self._document_vectors: torch.Tensor | None = None

    def _build_vocabulary(self, texts: Iterable[str]) -> None:
        terms = sorted({term for text in texts for term in tokenize(text)})
        self.vocabulary = {"<unk>": 0}
        for term in terms:
            if term not in self.vocabulary:
                self.vocabulary[term] = len(self.vocabulary)

    def _features(self, texts: list[str]) -> torch.Tensor:
        matrix = torch.zeros((len(texts), len(self.vocabulary)), dtype=torch.float32)
        for row, text in enumerate(texts):
            for term in tokenize(text):
                matrix[row, self.vocabulary.get(term, 0)] += 1.0
            total = matrix[row].sum()
            if total > 0:
                matrix[row] /= total
        return matrix

    def fit(
        self,
        documents: dict[str, str],
        pairs: list[TrainingPair],
        *,
        epochs: int = 400,
        learning_rate: float = 0.05,
        temperature: float = 0.08,
    ) -> list[float]:
        if not documents:
            raise ValueError("documents must not be empty")
        if not pairs:
            raise ValueError("pairs must not be empty")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        unknown_documents = {pair.document_id for pair in pairs} - set(documents)
        if unknown_documents:
            raise ValueError(f"training pairs reference unknown documents: {unknown_documents}")

        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)

        self.document_ids = sorted(documents)
        document_texts = [documents[document_id] for document_id in self.document_ids]
        query_texts = [pair.query for pair in pairs]
        self._build_vocabulary([*document_texts, *query_texts])

        self._model = _DualProjection(len(self.vocabulary), self.dimensions)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        query_features = self._features(query_texts)
        document_features = self._features(document_texts)
        document_position = {
            document_id: index for index, document_id in enumerate(self.document_ids)
        }
        labels = torch.tensor(
            [document_position[pair.document_id] for pair in pairs], dtype=torch.long
        )

        losses: list[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            query_vectors = self._model.encode_queries(query_features)
            document_vectors = self._model.encode_documents(document_features)
            logits = (query_vectors @ document_vectors.T) / temperature
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        with torch.no_grad():
            self._document_vectors = self._model.encode_documents(document_features)
        return losses

    def search(self, query: str, *, k: int = 5) -> list[tuple[str, float]]:
        if self._model is None or self._document_vectors is None:
            raise RuntimeError("fit must be called before search")
        if k <= 0:
            return []
        with torch.no_grad():
            query_vector = self._model.encode_queries(self._features([query]))[0]
            scores = self._document_vectors @ query_vector
        ranked = [
            (document_id, float(scores[index]))
            for index, document_id in enumerate(self.document_ids)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]
