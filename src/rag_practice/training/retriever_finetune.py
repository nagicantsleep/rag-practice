from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rag_practice.evaluation.training import (
    RankedDocument,
    RetrievalQuery,
    evaluate_rankings,
    select_top_non_positive,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1c82ace116a2629de82404c4be48c0e5d4cf08be"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    temperature: float
    max_sequence_length: int


@dataclass(frozen=True)
class BenchmarkSplit:
    documents: dict[str, str]
    queries: tuple[RetrievalQuery, ...]


@dataclass(frozen=True)
class TrainingBenchmark:
    config: TrainingConfig
    train: BenchmarkSplit
    dev: BenchmarkSplit
    test: BenchmarkSplit


def load_training_benchmark(path: str | Path) -> TrainingBenchmark:
    payload = json.loads(Path(path).read_text())
    model = payload["baseline_model"]
    if model["name"] != MODEL_NAME or model["revision"] != MODEL_REVISION:
        raise ValueError("benchmark baseline model does not match pinned implementation")

    contract = payload["training_contract"]
    config = TrainingConfig(
        seed=int(contract["seed"]),
        epochs=int(contract["epochs"]),
        batch_size=int(contract["batch_size"]),
        learning_rate=float(contract["learning_rate"]),
        temperature=float(contract["temperature"]),
        max_sequence_length=int(contract["max_sequence_length"]),
    )

    def parse_split(name: str) -> BenchmarkSplit:
        split = payload["splits"][name]
        documents = {str(key): str(value) for key, value in split["documents"].items()}
        queries = tuple(
            RetrievalQuery(
                id=str(item["id"]),
                query=str(item["query"]),
                relevant=str(item["relevant"]),
                query_class=str(item["class"]),
            )
            for item in split["queries"]
        )
        unknown = {query.relevant for query in queries} - set(documents)
        if unknown:
            raise ValueError(f"{name} qrels reference unknown documents: {sorted(unknown)}")
        return BenchmarkSplit(documents=documents, queries=queries)

    benchmark = TrainingBenchmark(
        config=config,
        train=parse_split("train"),
        dev=parse_split("dev"),
        test=parse_split("test"),
    )
    _validate_split_integrity(benchmark)
    return benchmark


def _validate_split_integrity(benchmark: TrainingBenchmark) -> None:
    train_documents = set(benchmark.train.documents)
    dev_documents = set(benchmark.dev.documents)
    test_documents = set(benchmark.test.documents)
    if train_documents & dev_documents or train_documents & test_documents or dev_documents & test_documents:
        raise ValueError("document ids must be disjoint across train/dev/test")

    train_queries = {query.id for query in benchmark.train.queries}
    dev_queries = {query.id for query in benchmark.dev.queries}
    test_queries = {query.id for query in benchmark.test.queries}
    if train_queries & dev_queries or train_queries & test_queries or dev_queries & test_queries:
        raise ValueError("query ids must be disjoint across train/dev/test")


def load_pinned_model(*, max_sequence_length: int) -> tuple[Any, float]:
    started = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("M10 retriever training requires the 'pretrained' extra") from exc

    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    model.max_seq_length = max_sequence_length
    return model, (time.perf_counter() - started) * 1000.0


def _parameter_stats(model: Any) -> dict[str, int]:
    parameters = list(model.parameters())
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    }


def rank_documents(
    model: Any,
    *,
    documents: dict[str, str],
    queries: Sequence[RetrievalQuery],
) -> tuple[dict[str, list[RankedDocument]], dict[str, float]]:
    import torch

    document_ids = sorted(documents)
    document_texts = [documents[document_id] for document_id in document_ids]

    model.eval()
    index_started = time.perf_counter()
    with torch.no_grad():
        document_embeddings = model.encode(
            document_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    index_build_ms = (time.perf_counter() - index_started) * 1000.0

    rankings: dict[str, list[RankedDocument]] = {}
    query_latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        with torch.no_grad():
            query_embedding = model.encode(
                [query.query],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            scores = document_embeddings @ query_embedding
        query_latencies.append((time.perf_counter() - started) * 1000.0)
        ranked = [
            RankedDocument(document_id=document_id, score=float(scores[index].item()))
            for index, document_id in enumerate(document_ids)
        ]
        ranked.sort(key=lambda item: (-item.score, item.document_id))
        rankings[query.id] = ranked

    return rankings, {
        "index_build_ms": index_build_ms,
        "mean_query_ms": sum(query_latencies) / len(query_latencies) if query_latencies else 0.0,
        "logical_index_bytes_float32": len(document_ids)
        * int(document_embeddings.shape[1])
        * 4,
    }


def mine_train_hard_negatives(
    baseline_model: Any,
    split: BenchmarkSplit,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    rankings, _ = rank_documents(
        baseline_model,
        documents=split.documents,
        queries=split.queries,
    )
    negatives: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for query in split.queries:
        ranking = rankings[query.id]
        negative = select_top_non_positive(ranking, positive_document_id=query.relevant)
        negative_rank = next(
            rank for rank, item in enumerate(ranking, start=1) if item.document_id == negative.document_id
        )
        positive_rank = next(
            rank for rank, item in enumerate(ranking, start=1) if item.document_id == query.relevant
        )
        negatives[query.id] = negative.document_id
        records.append(
            {
                "query_id": query.id,
                "positive_document_id": query.relevant,
                "positive_rank": positive_rank,
                "negative_document_id": negative.document_id,
                "negative_rank": negative_rank,
                "negative_score": negative.score,
            }
        )
    return negatives, records


def _differentiable_embeddings(model: Any, texts: Sequence[str]) -> Any:
    import torch.nn.functional as F

    features = model.tokenize(list(texts))
    device = next(model.parameters()).device
    features = {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in features.items()
    }
    embeddings = model(features)["sentence_embedding"]
    return F.normalize(embeddings, p=2, dim=-1)


def _train_model(
    model: Any,
    *,
    split: BenchmarkSplit,
    config: TrainingConfig,
    hard_negatives: dict[str, str] | None,
) -> tuple[list[float], float]:
    import torch
    import torch.nn.functional as F

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    queries = list(split.queries)
    loss_history: list[float] = []

    started = time.perf_counter()
    model.train()
    for epoch in range(config.epochs):
        indices = list(range(len(queries)))
        random.Random(config.seed + epoch).shuffle(indices)
        for offset in range(0, len(indices), config.batch_size):
            batch_indices = indices[offset : offset + config.batch_size]
            batch = [queries[index] for index in batch_indices]
            query_texts = [item.query for item in batch]
            positive_texts = [split.documents[item.relevant] for item in batch]

            optimizer.zero_grad(set_to_none=True)
            query_embeddings = _differentiable_embeddings(model, query_texts)
            positive_embeddings = _differentiable_embeddings(model, positive_texts)

            if hard_negatives is None:
                logits = (query_embeddings @ positive_embeddings.T) / config.temperature
            else:
                negative_texts = [split.documents[hard_negatives[item.id]] for item in batch]
                negative_embeddings = _differentiable_embeddings(model, negative_texts)
                candidates = torch.cat([positive_embeddings, negative_embeddings], dim=0)
                logits = (query_embeddings @ candidates.T) / config.temperature

            labels = torch.arange(len(batch), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu().item()))

    model.eval()
    training_ms = (time.perf_counter() - started) * 1000.0
    return loss_history, training_ms


def train_pair_only(
    split: BenchmarkSplit, config: TrainingConfig
) -> tuple[Any, dict[str, object]]:
    model, load_ms = load_pinned_model(max_sequence_length=config.max_sequence_length)
    loss_history, training_ms = _train_model(
        model,
        split=split,
        config=config,
        hard_negatives=None,
    )
    return model, {
        "model_load_ms": load_ms,
        "training_ms": training_ms,
        "loss_history": loss_history,
        **_parameter_stats(model),
    }


def train_with_hard_negatives(
    split: BenchmarkSplit,
    config: TrainingConfig,
    hard_negatives: dict[str, str],
) -> tuple[Any, dict[str, object]]:
    model, load_ms = load_pinned_model(max_sequence_length=config.max_sequence_length)
    loss_history, training_ms = _train_model(
        model,
        split=split,
        config=config,
        hard_negatives=hard_negatives,
    )
    return model, {
        "model_load_ms": load_ms,
        "training_ms": training_ms,
        "loss_history": loss_history,
        **_parameter_stats(model),
    }


def evaluate_model_on_split(model: Any, split: BenchmarkSplit) -> dict[str, object]:
    rankings, system = rank_documents(model, documents=split.documents, queries=split.queries)
    result = evaluate_rankings(split.queries, rankings)
    result["system"] = system
    return result


def describe_model(model: Any, *, model_load_ms: float) -> dict[str, object]:
    dimension = int(model.get_embedding_dimension())
    return {
        "name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "device": "cpu",
        "dtype": "float32",
        "dimensions": dimension,
        "model_load_ms": model_load_ms,
        **_parameter_stats(model),
    }
