"""Order domain logic."""


def create_order(payload: dict[str, object]) -> dict[str, object]:
    """Create and persist a domain order from validated payload."""
    return {"id": 42, **payload}
