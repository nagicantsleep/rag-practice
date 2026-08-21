"""Billing reference normalization."""


def normalize(value: str) -> str:
    """Normalize billing invoice references by removing spaces and uppercasing."""
    return value.replace(" ", "").upper()
