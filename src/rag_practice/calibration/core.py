"""Frozen M12 base RAG trace, observable features, and confidence methods."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")
ANSWER_RE = re.compile(r"ANSWER=([A-Z0-9_-]+)")
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "for",
    "of",
    "to",
    "in",
    "on",
    "and",
    "or",
    "does",
    "do",
    "with",
    "current",
    "please",
    "tell",
    "me",
}
FEATURE_NAMES = (
    "top1_score",
    "top2_score",
    "margin",
    "top1_valid",
    "top3_valid_fraction",
    "top3_entity_agreement",
    "answer_present",
    "answer_support",
    "conflict_signal",
    "retrieved_count",
)


@dataclass(frozen=True)
class Document:
    id: str
    entity_id: str
    text: str
    updated_generation: int
    trusted: bool
    active: bool

    @property
    def valid(self) -> bool:
        return self.active and self.trusted


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    entity_id: str
    question: str
    split: str
    scenario: str
    shift_class: str
    answerable: bool
    expected_answer: str | None
    required_evidence_ids: tuple[str, ...]
    forbidden_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationBenchmark:
    version: int
    seed: int
    documents: tuple[Document, ...]
    cases: tuple[BenchmarkCase, ...]

    def cases_for(self, split: str) -> tuple[BenchmarkCase, ...]:
        return tuple(case for case in self.cases if case.split == split)


@dataclass(frozen=True)
class RankedDocument:
    document: Document
    score: float


@dataclass(frozen=True)
class RuntimeTrace:
    query_id: str
    entity_id: str
    question: str
    retrieved: tuple[RankedDocument, ...]
    answer: str
    features: tuple[float, ...]

    def feature_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.features, strict=True))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.document.id for item in self.retrieved)


@dataclass(frozen=True)
class LogisticCalibrator:
    weights: tuple[float, ...]
    intercept: float

    def predict(self, features: Iterable[float]) -> float:
        z = self.intercept + sum(w * x for w, x in zip(self.weights, features, strict=True))
        return _sigmoid(z)

    @classmethod
    def fit(
        cls,
        rows: Iterable[tuple[tuple[float, ...], int]],
        *,
        epochs: int = 400,
        learning_rate: float = 0.10,
        l2: float = 0.01,
    ) -> "LogisticCalibrator":
        data = list(rows)
        if not data:
            raise ValueError("logistic calibrator requires at least one training row")
        width = len(data[0][0])
        if width != len(FEATURE_NAMES):
            raise ValueError("unexpected M12 feature width")
        weights = [0.0] * width
        intercept = 0.0
        n = float(len(data))
        for _ in range(epochs):
            grad_w = [0.0] * width
            grad_b = 0.0
            for features, target in data:
                if len(features) != width:
                    raise ValueError("inconsistent feature width")
                prediction = _sigmoid(intercept + sum(w * x for w, x in zip(weights, features, strict=True)))
                error = prediction - float(target)
                grad_b += error
                for index, value in enumerate(features):
                    grad_w[index] += error * value
            grad_b /= n
            for index in range(width):
                grad_w[index] = grad_w[index] / n + l2 * weights[index]
                weights[index] -= learning_rate * grad_w[index]
            intercept -= learning_rate * grad_b
        return cls(tuple(weights), intercept)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _content_tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS}


def _format_template(template: str, *, entity: str, answer: str, ordinal: int) -> str:
    return template.format(
        entity=entity,
        answer=answer,
        wrong=f"WRONG{ordinal:02d}",
        alt=f"ALT{ordinal:02d}",
        open=f"OPEN{ordinal:02d}",
        closed=f"CLOSED{ordinal:02d}",
    )


def load_benchmark(path: str | Path) -> CalibrationBenchmark:
    raw = json.loads(Path(path).read_text())
    generation = raw["document_generation"]
    templates = generation["templates"]
    documents: list[Document] = []
    cases: list[BenchmarkCase] = []
    for row in raw["queries"]:
        ordinal = int(row["id"][1:])
        template = templates[row["scenario"]]
        entity = row["entity_id"]
        answer = row["answer_token"]
        question = _format_template(template["question"], entity=entity, answer=answer, ordinal=ordinal)
        for suffix in ("a", "b", "c"):
            documents.append(
                Document(
                    id=f"{entity}-{suffix}",
                    entity_id=entity,
                    text=_format_template(template[suffix], entity=entity, answer=answer, ordinal=ordinal),
                    updated_generation=0,
                    trusted=True,
                    active=bool(template["a_active"]) if suffix == "a" else True,
                )
            )
        cases.append(
            BenchmarkCase(
                id=row["id"],
                entity_id=entity,
                question=question,
                split=row["split"],
                scenario=row["scenario"],
                shift_class=row["shift_class"],
                answerable=bool(row["answerable"]),
                expected_answer=row["expected_answer"],
                required_evidence_ids=tuple(row["required_evidence_ids"]),
                forbidden_evidence_ids=tuple(row["forbidden_evidence_ids"]),
            )
        )
    return CalibrationBenchmark(
        version=int(raw["version"]),
        seed=int(raw["seed"]),
        documents=tuple(documents),
        cases=tuple(cases),
    )


def retrieve(question: str, documents: Iterable[Document], *, top_k: int = 3) -> tuple[RankedDocument, ...]:
    query_tokens = _content_tokens(question)
    denominator = max(1, len(query_tokens))
    ranked = []
    for document in documents:
        score = len(query_tokens & _content_tokens(document.text)) / denominator
        ranked.append(RankedDocument(document=document, score=score))
    ranked.sort(key=lambda item: (-item.score, item.document.id))
    return tuple(ranked[:top_k])


def generate_answer(retrieved: tuple[RankedDocument, ...]) -> str:
    if not retrieved or retrieved[0].score == 0:
        return "UNKNOWN"
    match = ANSWER_RE.search(retrieved[0].document.text)
    return match.group(1) if match else "UNKNOWN"


def _conflict_signal(entity_id: str, retrieved: tuple[RankedDocument, ...]) -> float:
    values = set()
    for item in retrieved:
        document = item.document
        if not document.valid or document.entity_id != entity_id:
            continue
        match = ANSWER_RE.search(document.text)
        if match:
            values.add(match.group(1))
    return float(len(values) >= 2)


def build_runtime_trace(
    query_id: str,
    entity_id: str,
    question: str,
    documents: Iterable[Document],
) -> RuntimeTrace:
    retrieved = retrieve(question, documents, top_k=3)
    answer = generate_answer(retrieved)
    top1 = retrieved[0].score if retrieved else 0.0
    top2 = retrieved[1].score if len(retrieved) > 1 else 0.0
    valid_fraction = (
        sum(float(item.document.valid) for item in retrieved) / len(retrieved) if retrieved else 0.0
    )
    entity_counts = Counter(item.document.entity_id for item in retrieved)
    agreement = max(entity_counts.values()) / len(retrieved) if retrieved else 0.0
    top1_document = retrieved[0].document if retrieved else None
    answer_present = float(answer != "UNKNOWN")
    answer_support = float(bool(top1_document and answer != "UNKNOWN" and answer in top1_document.text))
    features = (
        top1,
        top2,
        top1 - top2,
        float(bool(top1_document and top1_document.valid)),
        valid_fraction,
        agreement,
        answer_present,
        answer_support,
        _conflict_signal(entity_id, retrieved),
        float(len(retrieved)),
    )
    return RuntimeTrace(
        query_id=query_id,
        entity_id=entity_id,
        question=question,
        retrieved=retrieved,
        answer=answer,
        features=features,
    )


def baseline_confidences(trace: RuntimeTrace) -> dict[str, float]:
    f = trace.feature_dict()
    return {
        "constant": 0.5,
        "top1": min(1.0, max(0.0, f["top1_score"])),
        "margin": min(1.0, max(0.0, f["margin"])),
        "hand_composed": min(
            1.0,
            max(
                0.0,
                0.45 * f["top1_score"]
                + 0.25 * f["margin"]
                + 0.15 * f["top1_valid"]
                + 0.10 * f["answer_present"]
                + 0.05 * f["answer_support"]
                - 0.25 * f["conflict_signal"],
            ),
        ),
    }
