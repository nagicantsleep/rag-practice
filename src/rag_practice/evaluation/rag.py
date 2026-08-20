from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_tokens(prediction)
    ref = normalize_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)

    remaining: dict[str, int] = {}
    for token in ref:
        remaining[token] = remaining.get(token, 0) + 1

    common = 0
    for token in pred:
        if remaining.get(token, 0) > 0:
            common += 1
            remaining[token] -= 1

    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer_contains_reference(prediction: str, reference: str) -> float:
    normalized_prediction = " ".join(normalize_tokens(prediction))
    normalized_reference = " ".join(normalize_tokens(reference))
    if not normalized_reference:
        return 0.0
    return float(normalized_reference in normalized_prediction)


def grounded_token_recall(answer: str, contexts: Iterable[str]) -> float:
    answer_tokens = normalize_tokens(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = set(normalize_tokens(" ".join(contexts)))
    return sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)


def citation_precision(cited_document_ids: Iterable[str], relevant_document_ids: set[str]) -> float:
    cited = list(cited_document_ids)
    if not cited:
        return 0.0
    return sum(document_id in relevant_document_ids for document_id in cited) / len(cited)


def citation_recall(cited_document_ids: Iterable[str], relevant_document_ids: set[str]) -> float:
    if not relevant_document_ids:
        return 0.0
    return len(set(cited_document_ids) & relevant_document_ids) / len(relevant_document_ids)
