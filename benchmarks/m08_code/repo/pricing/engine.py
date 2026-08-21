"""Checkout pricing engine."""
from pricing.discounts import apply_discount
from pricing.tax import tax_for_region


def compute_total(cart: list[int], region: str, coupon: str | None) -> int:
    """Compute final checkout total after discount and regional tax."""
    subtotal = sum(cart)
    discounted = apply_discount(subtotal, coupon)
    return discounted + tax_for_region(region, discounted)
