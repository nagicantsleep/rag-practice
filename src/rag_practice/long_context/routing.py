"""Transparent long-context vs retrieval routing controls for M08.7."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from rag_practice.ir.bm25 import BM25Index
from rag_practice.ir.text import tokenize

Route = Literal["direct", "retrieve"]


@dataclass(frozen=True)
class ContextSection:
    id: str
    text: str


@dataclass(frozen=True)
class ContextBundle:
    id: str
    title: str
    sections: tuple[ContextSection, ...]

    @property
    def word_count(self) -> int:
        return sum(len(tokenize(section.text)) for section in self.sections)


@dataclass(frozen=True)
class RoutingQuery:
    id: str
    bundle_id: str
    question: str
    task_class: str
    preferred_route: Route
    relevant: tuple[str, ...]
    expected_answer: str
    answer_kind: str


@dataclass(frozen=True)
class RoutingContract:
    retrieval_top_k: int
    direct_word_threshold: int
    global_route_markers: tuple[str, ...]
    abstain_token: str


@dataclass(frozen=True)
class LongContextBenchmark:
    contract: RoutingContract
    bundles: dict[str, ContextBundle]
    queries: tuple[RoutingQuery, ...]


@dataclass(frozen=True)
class ContextSelection:
    route: Route
    section_ids: tuple[str, ...]
    texts: tuple[str, ...]
    retrieval_scores: tuple[float, ...]
    retrieval_calls: int
    context_words: int
    full_context_words: int
    latency_ms: float


def _render_section(record: dict[str, object], template: str | None) -> ContextSection:
    section_id = str(record["id"])
    if "text" in record:
        text = str(record["text"])
    else:
        if template is None:
            raise ValueError(f"section {section_id} requires a template")
        special = str(record.get("special", ""))
        text = " ".join(template.format(special=special).split())
    return ContextSection(id=section_id, text=text)


def load_benchmark(path: str | Path) -> LongContextBenchmark:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = payload["frozen_contract"]
    templates = dict(frozen.get("templates", {}))
    contract = RoutingContract(
        retrieval_top_k=int(frozen["retrieval_top_k"]),
        direct_word_threshold=int(frozen["direct_word_threshold"]),
        global_route_markers=tuple(str(item).lower() for item in frozen["global_route_markers"]),
        abstain_token=str(frozen["abstain_token"]),
    )
    if contract.retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    if contract.direct_word_threshold <= 0:
        raise ValueError("direct_word_threshold must be positive")

    bundles: dict[str, ContextBundle] = {}
    for bundle_id, record in payload["bundles"].items():
        template_name = record.get("template")
        template = templates[str(template_name)] if template_name else None
        sections = tuple(_render_section(section, template) for section in record["sections"])
        if not sections:
            raise ValueError(f"bundle {bundle_id} must contain sections")
        if len({section.id for section in sections}) != len(sections):
            raise ValueError(f"bundle {bundle_id} contains duplicate section ids")
        bundle = ContextBundle(id=str(bundle_id), title=str(record["title"]), sections=sections)
        stored_words = int(record["word_count"])
        if bundle.word_count != stored_words:
            raise ValueError(
                f"bundle {bundle_id} word count changed: expected {stored_words}, got {bundle.word_count}"
            )
        bundles[bundle.id] = bundle

    queries: list[RoutingQuery] = []
    seen_query_ids: set[str] = set()
    for record in payload["queries"]:
        query = RoutingQuery(
            id=str(record["id"]),
            bundle_id=str(record["bundle_id"]),
            question=str(record["question"]),
            task_class=str(record["task_class"]),
            preferred_route=str(record["preferred_route"]),
            relevant=tuple(str(item) for item in record["relevant"]),
            expected_answer=str(record["expected_answer"]),
            answer_kind=str(record["answer_kind"]),
        )
        if query.id in seen_query_ids:
            raise ValueError(f"duplicate query id: {query.id}")
        seen_query_ids.add(query.id)
        if query.bundle_id not in bundles:
            raise ValueError(f"unknown bundle: {query.bundle_id}")
        if query.preferred_route not in {"direct", "retrieve"}:
            raise ValueError(f"invalid preferred route: {query.preferred_route}")
        section_ids = {section.id for section in bundles[query.bundle_id].sections}
        unknown = set(query.relevant) - section_ids
        if unknown:
            raise ValueError(f"query {query.id} references unknown sections: {sorted(unknown)}")
        queries.append(query)

    return LongContextBenchmark(contract=contract, bundles=bundles, queries=tuple(queries))


class ExplicitLongContextRouter:
    """Qrel-blind router using only bundle size and frozen query-language markers."""

    def __init__(self, contract: RoutingContract) -> None:
        self.contract = contract

    def route(self, question: str, bundle_word_count: int) -> Route:
        lowered = question.lower()
        if bundle_word_count <= self.contract.direct_word_threshold:
            return "direct"
        if any(marker in lowered for marker in self.contract.global_route_markers):
            return "direct"
        return "retrieve"


def select_context(
    benchmark: LongContextBenchmark,
    query: RoutingQuery,
    *,
    route: Route,
) -> ContextSelection:
    bundle = benchmark.bundles[query.bundle_id]
    start = perf_counter()
    if route == "direct":
        selected = list(bundle.sections)
        scores = [0.0] * len(selected)
        calls = 0
    elif route == "retrieve":
        documents = {section.id: section.text for section in bundle.sections}
        index = BM25Index(documents)
        ranked = index.search(query.question, k=benchmark.contract.retrieval_top_k)
        by_id = {section.id: section for section in bundle.sections}
        selected = [by_id[section_id] for section_id, _ in ranked]
        scores = [score for _, score in ranked]
        calls = 1
    else:
        raise ValueError(f"unsupported route: {route}")
    elapsed_ms = (perf_counter() - start) * 1000.0
    return ContextSelection(
        route=route,
        section_ids=tuple(section.id for section in selected),
        texts=tuple(section.text for section in selected),
        retrieval_scores=tuple(float(score) for score in scores),
        retrieval_calls=calls,
        context_words=sum(len(tokenize(section.text)) for section in selected),
        full_context_words=bundle.word_count,
        latency_ms=elapsed_ms,
    )


class DeterministicEvidenceReader:
    """Qrel-blind teaching reader that derives answers only from selected text."""

    def __init__(self, *, abstain_token: str = "ABSTAIN") -> None:
        self.abstain_token = abstain_token

    def answer(self, question: str, texts: tuple[str, ...] | list[str]) -> str:
        context = "\n".join(texts)
        lowered = question.lower()

        patterns: tuple[tuple[str, str], ...] = (
            ("service desk hotline", r"Cedar service desk hotline:\s*([0-9-]+)"),
            ("shipping window", r"Cedar shipping window closes at\s*([0-9:]+\s*UTC)"),
            ("emergency beacon frequency", r"Atlas emergency beacon frequency:\s*([0-9.]+\s*MHz)"),
            ("rollback phrase", r'Orion rollback phrase:\s*"([^"]+)"'),
            ("audit phrase", r"Lumen audit phrase:\s*([a-z]+\s+[a-z]+)"),
        )
        for cue, pattern in patterns:
            if cue in lowered:
                match = re.search(pattern, context, re.IGNORECASE)
                return match.group(1).strip() if match else self.abstain_token

        if "mandatory checks" in lowered or "mandatory checkpoints" in lowered:
            count = len(re.findall(r"Mandatory check(?:point)?\s*[—-]\s*", context, re.IGNORECASE))
            return str(count) if count else self.abstain_token

        if "escalation contact" in lowered:
            values = re.findall(
                r"Escalation contact\s*[—-]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
                context,
            )
            return "; ".join(values) if values else self.abstain_token

        if "higher reserve" in lowered:
            pairs = re.findall(
                r"Orion (North|South) reserve:\s*(\d+)\s*units",
                context,
                re.IGNORECASE,
            )
            reserves = {site.lower(): int(value) for site, value in pairs}
            if {"north", "south"} <= reserves.keys():
                return "North" if reserves["north"] > reserves["south"] else "South"
            return self.abstain_token

        if "release code" in lowered:
            values = re.findall(r"Orion release code:\s*([A-Z]+-\d+)", context)
            return "; ".join(values) if values else self.abstain_token

        return self.abstain_token
