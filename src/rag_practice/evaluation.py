import json
import re
from dataclasses import dataclass
from pathlib import Path

from .store import SearchResult, VectorStore


@dataclass(frozen=True)
class RelevantPassage:
    source: str
    contains: str


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    relevant_passages: tuple[RelevantPassage, ...]


@dataclass(frozen=True)
class ExampleResult:
    id: str
    question: str
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    matched_passages: int
    total_relevant_passages: int


@dataclass(frozen=True)
class RetrievalReport:
    top_k: int
    examples: tuple[ExampleResult, ...]

    @property
    def recall_at_k(self) -> float:
        if not self.examples:
            return 0.0
        return sum(example.recall_at_k for example in self.examples) / len(self.examples)

    @property
    def mrr(self) -> float:
        if not self.examples:
            return 0.0
        return sum(example.reciprocal_rank for example in self.examples) / len(self.examples)

    @property
    def hit_rate_at_k(self) -> float:
        if not self.examples:
            return 0.0
        hits = sum(example.first_relevant_rank is not None for example in self.examples)
        return hits / len(self.examples)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _source_matches(result_source: str, expected_source: str) -> bool:
    result = Path(result_source).as_posix()
    expected = Path(expected_source).as_posix()
    return result == expected or result.endswith(f"/{expected}")


def result_matches_passage(result: SearchResult, passage: RelevantPassage) -> bool:
    return _source_matches(result.source, passage.source) and _normalize_text(
        passage.contains
    ) in _normalize_text(result.text)


def evaluate_results(
    example: EvalExample,
    results: list[SearchResult],
) -> ExampleResult:
    if not example.relevant_passages:
        raise ValueError(f"Evaluation example {example.id!r} has no relevant passages")

    matched_passages: set[int] = set()
    first_relevant_rank: int | None = None

    for rank, result in enumerate(results, start=1):
        result_is_relevant = False
        for passage_index, passage in enumerate(example.relevant_passages):
            if result_matches_passage(result, passage):
                matched_passages.add(passage_index)
                result_is_relevant = True

        if result_is_relevant and first_relevant_rank is None:
            first_relevant_rank = rank

    recall = len(matched_passages) / len(example.relevant_passages)
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

    return ExampleResult(
        id=example.id,
        question=example.question,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        first_relevant_rank=first_relevant_rank,
        matched_passages=len(matched_passages),
        total_relevant_passages=len(example.relevant_passages),
    )


def load_eval_dataset(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []

    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                item = json.loads(line)
                relevant_passages = tuple(
                    RelevantPassage(
                        source=str(passage["source"]),
                        contains=str(passage["contains"]),
                    )
                    for passage in item["relevant_passages"]
                )
                example = EvalExample(
                    id=str(item.get("id", f"line-{line_number}")),
                    question=str(item["question"]),
                    relevant_passages=relevant_passages,
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid evaluation example at {path}:{line_number}: {exc}"
                ) from exc

            if not example.question.strip():
                raise ValueError(f"Empty question at {path}:{line_number}")
            if not example.relevant_passages:
                raise ValueError(f"No relevant passages at {path}:{line_number}")
            examples.append(example)

    if not examples:
        raise ValueError(f"Evaluation dataset is empty: {path}")

    return examples


def evaluate_retrieval(
    store: VectorStore,
    examples: list[EvalExample],
    top_k: int,
) -> RetrievalReport:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    results = tuple(
        evaluate_results(example, store.search(example.question, top_k=top_k))
        for example in examples
    )
    return RetrievalReport(top_k=top_k, examples=results)


def format_report(report: RetrievalReport, show_misses: bool = False) -> str:
    lines = [
        "=== RETRIEVAL EVALUATION ===",
        f"questions: {len(report.examples)}",
        f"top_k: {report.top_k}",
        f"Recall@{report.top_k}: {report.recall_at_k:.3f}",
        f"MRR: {report.mrr:.3f}",
        f"HitRate@{report.top_k}: {report.hit_rate_at_k:.3f}",
    ]

    if show_misses:
        misses = [
            example for example in report.examples if example.first_relevant_rank is None
        ]
        lines.append("")
        lines.append("=== MISSES ===")
        if not misses:
            lines.append("None")
        else:
            for example in misses:
                lines.append(f"- {example.id}: {example.question}")

    return "\n".join(lines)
