"""Checkout orchestration service."""
from payments.gateway import charge
from pricing.engine import compute_total


def finalize_checkout(cart: list[int], region: str, coupon: str | None) -> str:
    """Finalize checkout by calling compute_total before payment charge."""
    total = compute_total(cart, region, coupon)
    return charge(total)
