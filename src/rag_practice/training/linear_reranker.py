from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rag_practice.ir.bm25 import BM25Index
from rag_practice.ir.text import tokenize
from rag_practice.training.retriever_finetune import (
    BenchmarkSplit,
    RankedDocument,
    RetrievalQuery,
    rank_documents,
)


@dataclass(frozen=True)
class RerankerConfig:
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float
    candidate_k: int


@dataclass(frozen=True)
class FeatureRow:
    dense_score: float
    bm25_score: float
    overlap_fraction: float
    reciprocal_rank: float

    def values(self) -> tuple[float, float, float, float]:
        return (
            self.dense_score,
            self.bm25_score,
            self.overlap_fraction,
            self.reciprocal_rank,
        )


def load_reranker_contract(path: str | Path) -> RerankerConfig:
    payload = json.loads(Path(path).read_text())
    training = payload["training"]
    generator = payload["candidate_generator"]
    expected_features = [
        "dense_cosine_similarity",
        "bm25_raw_score",
        "query_token_overlap_fraction",
        "reciprocal_first_stage_rank",
    ]
    if payload["features"] != expected_features:
        raise ValueError("unexpected reranker feature contract")
    if training["loss"] != "pairwise_softplus" or training["optimizer"] != "Adam":
        raise ValueError("unexpected reranker optimization contract")
    return RerankerConfig(
        seed=int(training["seed"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        candidate_k=int(generator["candidate_k"]),
    )


def _overlap_fraction(query: str, document: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    document_terms = set(tokenize(document))
    return len(query_terms & document_terms) / len(query_terms)


def build_candidate_features(
    *,
    query: RetrievalQuery,
    documents: dict[str, str],
    baseline_ranking: Sequence[RankedDocument],
    candidate_k: int,
) -> dict[str, FeatureRow]:
    bm25 = BM25Index(documents)
    rows: dict[str, FeatureRow] = {}
    for rank, candidate in enumerate(baseline_ranking[:candidate_k], start=1):
        rows[candidate.document_id] = FeatureRow(
            dense_score=candidate.score,
            bm25_score=bm25.score(query.query, candidate.document_id),
            overlap_fraction=_overlap_fraction(query.query, documents[candidate.document_id]),
            reciprocal_rank=1.0 / rank,
        )
    return rows


class LinearPairwiseReranker:
    def __init__(self, config: RerankerConfig) -> None:
        import torch

        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        self.config = config
        self.model = torch.nn.Linear(4, 1, bias=True)
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()
        self.loss_history: list[float] = []
        self.training_ms = 0.0
        self.training_pair_count = 0

    def fit(
        self,
        *,
        baseline_model: Any,
        split: BenchmarkSplit,
    ) -> None:
        import torch
        import torch.nn.functional as F

        rankings, _ = rank_documents(
            baseline_model, documents=split.documents, queries=split.queries
        )
        pairs: list[tuple[FeatureRow, FeatureRow]] = []
        for query in split.queries:
            candidates = rankings[query.id][: self.config.candidate_k]
            if query.relevant not in {item.document_id for item in candidates}:
                continue
            rows = build_candidate_features(
                query=query,
                documents=split.documents,
                baseline_ranking=rankings[query.id],
                candidate_k=self.config.candidate_k,
            )
            positive = rows[query.relevant]
            for candidate in candidates:
                if candidate.document_id != query.relevant:
                    pairs.append((positive, rows[candidate.document_id]))

        if not pairs:
            raise ValueError("reranker training produced no positive-negative pairs")
        self.training_pair_count = len(pairs)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        started = time.perf_counter()
        self.model.train()
        for _ in range(self.config.epochs):
            optimizer.zero_grad(set_to_none=True)
            positive_tensor = torch.tensor(
                [positive.values() for positive, _ in pairs], dtype=torch.float32
            )
            negative_tensor = torch.tensor(
                [negative.values() for _, negative in pairs], dtype=torch.float32
            )
            positive_scores = self.model(positive_tensor).squeeze(-1)
            negative_scores = self.model(negative_tensor).squeeze(-1)
            loss = F.softplus(-(positive_scores - negative_scores)).mean()
            loss.backward()
            optimizer.step()
            self.loss_history.append(float(loss.detach().item()))
        self.training_ms = (time.perf_counter() - started) * 1000.0
        self.model.eval()

    def score_rows(self, rows: Sequence[FeatureRow]) -> list[float]:
        import torch

        if not rows:
            return []
        tensor = torch.tensor([row.values() for row in rows], dtype=torch.float32)
        with torch.no_grad():
            values = self.model(tensor).squeeze(-1)
        return [float(value.item()) for value in values]

    def parameters_payload(self) -> dict[str, object]:
        weight = self.model.weight.detach().squeeze(0).tolist()
        bias = float(self.model.bias.detach().item())
        return {
            "weights": {
                "dense_cosine_similarity": float(weight[0]),
                "bm25_raw_score": float(weight[1]),
                "query_token_overlap_fraction": float(weight[2]),
                "reciprocal_first_stage_rank": float(weight[3]),
            },
            "bias": bias,
            "parameter_count": 5,
            "parameter_bytes_float32": 20,
        }


def evaluate_reranker(
    *,
    reranker: LinearPairwiseReranker,
    baseline_model: Any,
    split: BenchmarkSplit,
) -> dict[str, object]:
    rankings, baseline_system = rank_documents(
        baseline_model, documents=split.documents, queries=split.queries
    )
    candidate_hits = 0
    hit_at_1 = 0
    reciprocal_ranks = 0.0
    margins: list[float] = []
    query_latencies: list[float] = []
    per_query: list[dict[str, object]] = []

    for query in split.queries:
        baseline = rankings[query.id]
        candidates = baseline[: reranker.config.candidate_k]
        candidate_ids = [item.document_id for item in candidates]
        in_candidates = query.relevant in candidate_ids
        candidate_hits += int(in_candidates)

        rows = build_candidate_features(
            query=query,
            documents=split.documents,
            baseline_ranking=baseline,
            candidate_k=reranker.config.candidate_k,
        )
        started = time.perf_counter()
        scores = reranker.score_rows([rows[item.document_id] for item in candidates])
        query_latencies.append((time.perf_counter() - started) * 1000.0)
        reranked = sorted(
            zip(candidates, scores), key=lambda item: (-item[1], item[0].document_id)
        )
        reranked_ids = [item.document_id for item, _ in reranked]

        if in_candidates:
            rerank_position = reranked_ids.index(query.relevant) + 1
            reciprocal_ranks += 1.0 / rerank_position
            hit_at_1 += int(rerank_position == 1)
            relevant_score = next(
                score for item, score in reranked if item.document_id == query.relevant
            )
            best_negative = max(
                score for item, score in reranked if item.document_id != query.relevant
            )
            margins.append(relevant_score - best_negative)
        else:
            original_position = next(
                rank
                for rank, item in enumerate(baseline, start=1)
                if item.document_id == query.relevant
            )
            reciprocal_ranks += 1.0 / original_position

        per_query.append(
            {
                "id": query.id,
                "class": query.query_class,
                "relevant": query.relevant,
                "candidate_ids": candidate_ids,
                "candidate_contains_relevant": in_candidates,
                "reranked_ids": reranked_ids,
                "rerank_scores": [
                    {"document_id": item.document_id, "score": score}
                    for item, score in reranked
                ],
            }
        )

    count = len(split.queries)
    return {
        "candidate_recall@3": candidate_hits / count if count else 0.0,
        "recall@1": hit_at_1 / count if count else 0.0,
        "mrr": reciprocal_ranks / count if count else 0.0,
        "mean_rerank_margin": sum(margins) / len(margins) if margins else 0.0,
        "mean_rerank_ms": sum(query_latencies) / len(query_latencies)
        if query_latencies
        else 0.0,
        "baseline_system": baseline_system,
        "per_query": per_query,
    }
