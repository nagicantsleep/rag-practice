"""Payment gateway."""


def charge(amount: int) -> str:
    """Charge the payment gateway for a checkout amount."""
    return f"charged:{amount}"
