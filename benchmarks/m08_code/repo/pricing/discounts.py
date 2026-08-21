"""Checkout discount rules."""


def apply_discount(subtotal: int, coupon: str | None) -> int:
    """Apply a checkout coupon discount to a subtotal."""
    if coupon == "SAVE10":
        return subtotal * 90 // 100
    return subtotal
