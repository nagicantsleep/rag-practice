from __future__ import annotations

from collections.abc import Mapping

_ADJECTIVES = (
    "amber",
    "coastal",
    "alpine",
    "quiet",
    "ceramic",
    "winter",
    "copper",
    "meadow",
    "ancient",
    "silver",
)
_NOUNS = (
    "orchid",
    "basalt",
    "violin",
    "saffron",
    "glacier",
    "pottery",
    "cedar",
    "falcon",
    "lantern",
    "mosaic",
)
_VERBS = (
    "blooms",
    "weathers",
    "resonates",
    "ripens",
    "drifts",
    "hardens",
    "grows",
    "circles",
    "glows",
    "fades",
)
_CONTEXTS = (
    "before sunrise",
    "near a stone terrace",
    "during the autumn season",
    "beside a shallow creek",
    "inside a craft workshop",
    "along a mountain trail",
    "under a wooden shelter",
    "across a dry plateau",
    "within a garden courtyard",
    "after an evening storm",
)


def generate_distractors(count: int, *, start: int = 0) -> dict[str, str]:
    """Create deterministic, intentionally off-topic documents for scale tests.

    These distractors enlarge the candidate set without pretending to be a
    natural-language benchmark. Their vocabulary is deliberately unrelated to
    the IR/RAG target documents so the experiment measures robustness and
    retrieval-system scaling, not domain generalization.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if start < 0:
        raise ValueError("start must be non-negative")

    documents: dict[str, str] = {}
    for offset in range(count):
        index = start + offset
        adjective = _ADJECTIVES[index % len(_ADJECTIVES)]
        noun = _NOUNS[(index // 3) % len(_NOUNS)]
        verb = _VERBS[(index // 7) % len(_VERBS)]
        context = _CONTEXTS[(index // 11) % len(_CONTEXTS)]
        document_id = f"scale-{index:05d}"
        documents[document_id] = (
            f"The {adjective} {noun} {verb} {context}. "
            f"Catalogue specimen number {index} records color, texture, and season."
        )
    return documents


def expand_corpus(
    base_documents: Mapping[str, str],
    total_documents: int,
) -> dict[str, str]:
    """Return the base corpus plus enough deterministic distractors to hit size."""

    if total_documents < len(base_documents):
        raise ValueError("total_documents cannot be smaller than the base corpus")
    expanded = dict(base_documents)
    expanded.update(generate_distractors(total_documents - len(expanded)))
    if len(expanded) != total_documents:
        raise RuntimeError("generated distractor IDs collided with base document IDs")
    return expanded
