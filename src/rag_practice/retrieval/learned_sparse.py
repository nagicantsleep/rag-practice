from __future__ import annotations

from collections.abc import Iterable

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("LearnedSparseRetriever requires PyTorch") from exc

from rag_practice.ir.text import tokenize
from rag_practice.retrieval.neural_dual_encoder import TrainingPair


class _SparseExpansionModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.query_expansion = nn.Linear(vocab_size, vocab_size, bias=False)
        self.document_expansion = nn.Linear(vocab_size, vocab_size, bias=False)
        with torch.no_grad():
            self.query_expansion.weight.copy_(torch.eye(vocab_size))
            self.document_expansion.weight.copy_(torch.eye(vocab_size))

    @staticmethod
    def activate(logits: torch.Tensor) -> torch.Tensor:
        return torch.log1p(F.relu(logits))

    def encode_queries(self, features: torch.Tensor) -> torch.Tensor:
        return self.activate(self.query_expansion(features))

    def encode_documents(self, features: torch.Tensor) -> torch.Tensor:
        return self.activate(self.document_expansion(features))


class LearnedSparseRetriever:
    """Educational SPLADE-style learned sparse lexical expansion.

    The representation keeps explicit vocabulary coordinates, non-negative
    log-saturated weights, dot-product retrieval, and sparsity pressure. It
    intentionally replaces SPLADE's pretrained Transformer/MLM backbone with a
    transparent bag-of-words linear expansion layer so the mechanism is visible.
    """

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.vocabulary: dict[str, int] = {"<unk>": 0}
        self.terms: list[str] = ["<unk>"]
        self.document_ids: list[str] = []
        self._model: _SparseExpansionModel | None = None
        self._document_vectors: torch.Tensor | None = None

    def _build_vocabulary(self, texts: Iterable[str]) -> None:
        terms = sorted({term for text in texts for term in tokenize(text)})
        self.terms = ["<unk>", *terms]
        self.vocabulary = {term: index for index, term in enumerate(self.terms)}

    def _features(self, texts: list[str]) -> torch.Tensor:
        matrix = torch.zeros((len(texts), len(self.vocabulary)), dtype=torch.float32)
        for row, text in enumerate(texts):
            for term in tokenize(text):
                matrix[row, self.vocabulary.get(term, 0)] += 1.0
        return matrix

    def fit(
        self,
        documents: dict[str, str],
        pairs: list[TrainingPair],
        *,
        epochs: int = 300,
        learning_rate: float = 0.05,
        temperature: float = 0.1,
        sparsity_lambda: float = 0.01,
    ) -> list[float]:
        if not documents or not pairs:
            raise ValueError("documents and pairs must not be empty")
        if epochs <= 0 or learning_rate <= 0 or temperature <= 0:
            raise ValueError("epochs, learning_rate, and temperature must be positive")
        if sparsity_lambda < 0:
            raise ValueError("sparsity_lambda must be non-negative")

        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.document_ids = sorted(documents)
        document_texts = [documents[document_id] for document_id in self.document_ids]
        query_texts = [pair.query for pair in pairs]
        self._build_vocabulary([*document_texts, *query_texts])
        self._model = _SparseExpansionModel(len(self.vocabulary))

        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        query_features = self._features(query_texts)
        document_features = self._features(document_texts)
        positions = {document_id: index for index, document_id in enumerate(self.document_ids)}
        labels = torch.tensor(
            [positions[pair.document_id] for pair in pairs], dtype=torch.long
        )

        losses: list[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            query_vectors = self._model.encode_queries(query_features)
            document_vectors = self._model.encode_documents(document_features)
            scores = query_vectors @ document_vectors.T / temperature
            ranking_loss = F.cross_entropy(scores, labels)
            sparse_penalty = query_vectors.mean() + document_vectors.mean()
            loss = ranking_loss + sparsity_lambda * sparse_penalty
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        with torch.no_grad():
            self._document_vectors = self._model.encode_documents(document_features)
        return losses

    def encode_query(self, text: str) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("fit must be called before encode_query")
        with torch.no_grad():
            vector = self._model.encode_queries(self._features([text]))[0]
        return {
            self.terms[index]: float(value)
            for index, value in enumerate(vector)
            if float(value) > 1e-6
        }

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

    def mean_nonzero_dimensions(self, threshold: float = 1e-4) -> float:
        if self._document_vectors is None:
            raise RuntimeError("fit must be called first")
        return float(
            (self._document_vectors > threshold).sum(dim=1).float().mean()
        )
