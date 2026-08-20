"""Minimal text processing for the IR fundamentals milestone."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into simple alphanumeric tokens.

    This deliberately small tokenizer keeps M00 focused on retrieval math.
    Later milestones can compare richer tokenization and preprocessing.
    """

    return _TOKEN_RE.findall(text.lower())
