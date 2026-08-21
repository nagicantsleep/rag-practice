"""Order API wrapper."""
from domain.orders import create_order


def submit_order(payload: dict[str, object]) -> dict[str, object]:
    """Submit an API order by calling the domain create_order implementation."""
    return create_order(payload)
