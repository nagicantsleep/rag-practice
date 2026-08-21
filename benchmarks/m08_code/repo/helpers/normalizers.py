"""General text normalization."""


def normalize(value: str) -> str:
    """Normalize generic user text by trimming and lowercasing."""
    return value.strip().lower()
